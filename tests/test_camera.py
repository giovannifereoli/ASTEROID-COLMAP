import numpy as np
import pytest

from asteroid_colmap.catalog import pixel_frames
from asteroid_colmap.preprocess import FLIPS, apply_flip


def test_colmap_parameters(camera):
    fx, fy, cx, cy, k1, k2, p1, p2 = camera.colmap_params
    assert fx == pytest.approx(10716.2, abs=0.1)
    assert fy == pytest.approx(10723.1, abs=0.1)
    assert (cx, cy) == (512.0, 512.0)
    assert k1 == pytest.approx(0.1892, abs=1e-4)
    assert (k2, p1, p2) == (0.0, 0.0, 0.0)
    assert camera.colmap_model == "OPENCV"


def test_project_and_pixel_rays_round_trip(camera):
    u, v = np.meshgrid(np.linspace(0, 1024, 9), np.linspace(0, 1024, 9))
    rays = camera.pixel_rays(u, v)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0)
    uv = camera.project(rays.reshape(-1, 3))
    np.testing.assert_allclose(uv[:, 0], u.ravel(), atol=1e-6)
    np.testing.assert_allclose(uv[:, 1], v.ravel(), atol=1e-6)


def test_principal_point_is_boresight(camera):
    np.testing.assert_allclose(camera.pixel_rays(512.0, 512.0), [0.0, 0.0, 1.0])


@pytest.mark.parametrize("flip", FLIPS)
def test_pixel_frames_invert_the_image_flip(camera, flip):
    """A keypoint in the flipped PNG must index the same pixel in the stored FITS array."""
    fits = np.arange(camera.height * camera.width).reshape(camera.height, camera.width)
    png = apply_flip(fits, flip)
    rows = np.array([0, 17, 600, 1023])
    cols = np.array([5, 1023, 300, 0])
    # COLMAP puts the centre of pixel (row, col) at (u, v) = (col + 0.5, row + 0.5)
    f = pixel_frames(cols + 0.5, rows + 0.5, flip, camera)
    np.testing.assert_array_equal(fits[f["fits_row"].astype(int), f["fits_col"].astype(int)],
                                  png[rows, cols])
    np.testing.assert_allclose(f["sample_ik"], cols)
    np.testing.assert_allclose(f["line_ik"], rows)


def test_apply_flip_rejects_unknown():
    with pytest.raises(ValueError):
        apply_flip(np.zeros((2, 2)), "transpose")
