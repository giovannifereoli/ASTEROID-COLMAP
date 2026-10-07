import numpy as np
import pandas as pd
import pytest

from asteroid_colmap.catalog import compare_catalogs, equal_area_cells, grade, one_observation_per_image
from asteroid_colmap.model_io import SparseModel


def test_grade_thresholds():
    track = np.array([8, 8, 4, 4, 3, 20])
    angle = np.array([15.0, 14.9, 5.0, 4.9, 30.0, 40.0])
    error = np.array([1.0, 1.0, 2.0, 2.0, 0.1, 2.1])
    assert grade(track, angle, error).tolist() == ["A", "B", "B", "C", "C", "C"]


def test_equal_area_cells():
    R = 262.7
    lat = np.array([0.0, 0.0, 0.0, 89.9, 89.9])
    lon = np.array([0.0, 1.0, 180.0, 0.0, 180.0])
    cells = equal_area_cells(lat, lon, 10.0, R)
    assert cells[0] == cells[1]  # 1 deg of longitude ~ 4.6 km at the equator
    assert cells[0] != cells[2]
    assert cells[3] == cells[4]  # the polar cap is a single cell
    # roughly one cell per (10 km)^2 over the whole sphere
    lat_g, lon_g = np.meshgrid(np.linspace(-89.95, 89.95, 1800), np.linspace(0, 359.9, 3600))
    n = len(np.unique(equal_area_cells(lat_g.ravel(), lon_g.ravel(), 10.0, R)))
    assert n == pytest.approx(4 * np.pi * R**2 / 100.0, rel=0.02)


def _write_catalog(d, landmarks, observations):
    d.mkdir()
    pd.DataFrame(landmarks).to_csv(d / "landmarks.csv", index=False)
    pd.DataFrame(observations, columns=["landmark_id", "image", "u", "v"]).to_csv(
        d / "observations.csv", index=False)


def test_compare_catalogs(tmp_path):
    rng = np.random.default_rng(1)
    xyz = rng.normal(scale=200.0, size=(4, 3))
    offset = np.array([0.12, -0.05, 0.0])
    imgs = [f"im{k}.png" for k in range(5)]
    uv = rng.uniform(0, 1024, size=(4, 5, 2))

    obs_a = [(f"A{i}", imgs[k], *uv[i, k]) for i in range(4) for k in range(5)]
    # catalog B: same keypoints under other ids, shifted landmarks;
    # landmark 3 only shares two keypoints and must stay unmatched
    obs_b = [(f"B{3 - i}", imgs[k], *uv[i, k]) for i in range(3) for k in range(5)]
    obs_b += [("B9", imgs[k], *uv[3, k]) for k in range(2)]
    obs_b += [("B9", imgs[k], *(uv[3, k] + 50)) for k in range(2, 5)]
    lm = lambda ids, X: {"landmark_id": ids, "x_km": X[:, 0], "y_km": X[:, 1], "z_km": X[:, 2],
                         "height_km": np.linalg.norm(X, axis=1) - 260.0}
    _write_catalog(tmp_path / "a", lm([f"A{i}" for i in range(4)], xyz), obs_a)
    xb = np.vstack([xyz[2::-1] + offset, [[0.0, 0.0, 300.0]]])
    _write_catalog(tmp_path / "b", lm(["B1", "B2", "B3", "B9"], xb), obs_b)

    r = compare_catalogs(tmp_path / "a", tmp_path / "b")
    assert r["num_landmarks"] == [4, 4]
    assert r["num_matched"] == 3
    np.testing.assert_allclose(r["mean_offset_km"], -offset, atol=1e-9)
    np.testing.assert_allclose(r["distance_km_p50_p90_p99"], np.linalg.norm(offset), atol=1e-9)
    np.testing.assert_allclose(r["distance_minus_offset_km_p50_p90"], 0.0, atol=1e-9)
    assert compare_catalogs(tmp_path / "a", tmp_path / "b", min_shared=6)["num_matched"] == 0


def test_one_observation_per_image_keeps_closest_keypoint(camera):
    """Two nearby keypoints of one image in one track count once: the one nearer the
    projection stays, and only that point's track length and error are recomputed."""
    R = np.eye(3)
    images = pd.DataFrame({"image_id": [1, 2], "R": [R, R], "tx": 0.0, "ty": 0.0, "tz": [5000.0, 6000.0]})
    points = pd.DataFrame({"point3d_id": [7, 8], "x": [0.0, 10.0], "y": [0.0, -5.0], "z": 0.0,
                           "error": [9.0, 9.0], "track_length": [3, 2]})
    uv = {(p, i): camera.project(points.loc[points.point3d_id == p, ["x", "y", "z"]].to_numpy()
                                 + [0.0, 0.0, 5000.0 + 1000.0 * (i - 1)])[0]
          for p in (7, 8) for i in (1, 2)}
    rows = [(7, 1, uv[7, 1] + [2.4, 0.0]),   # farther copy, listed first
            (8, 1, uv[8, 1]),
            (7, 1, uv[7, 1] + [0.0, 0.5]),   # nearer copy
            (7, 2, uv[7, 2] + [0.3, 0.4]),
            (8, 2, uv[8, 2])]
    obs = pd.DataFrame([(p, i, k, u, v) for k, (p, i, (u, v)) in enumerate(rows)],
                       columns=["point3d_id", "image_id", "point2d_idx", "u", "v"])
    model = SparseModel(pd.DataFrame(), images, points, obs)

    pts, kept = one_observation_per_image(model, camera)
    assert kept.point2d_idx.tolist() == [1, 2, 3, 4]
    assert pts.track_length.tolist() == [2, 2]
    assert pts.error.tolist() == pytest.approx([0.5, 9.0], abs=1e-6)  # (0.5 + 0.5) / 2; 8 untouched
    assert model.points.track_length.tolist() == [3, 2]  # input not modified
