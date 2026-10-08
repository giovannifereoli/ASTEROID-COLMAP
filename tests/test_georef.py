import numpy as np
from scipy.spatial.transform import Rotation

from asteroid_colmap.georef import pointing_offsets, position_components

rng = np.random.default_rng(1)


def look_at(C):
    z = -C / np.linalg.norm(C)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def test_position_components_split_the_residual():
    C_ref = rng.normal(size=(20, 3)) * 5000.0
    d = rng.normal(size=(20, 3))
    split = position_components(C_ref + d, C_ref)
    np.testing.assert_allclose(np.linalg.norm(split, axis=1), np.linalg.norm(d, axis=1))
    up = C_ref / np.linalg.norm(C_ref, axis=1, keepdims=True)
    np.testing.assert_allclose(split[:, 0], (d * up).sum(1))
    # moving straight away from the body is pure range; a move along the pole above the
    # equator is pure north
    np.testing.assert_allclose(position_components(1.002 * C_ref[0], C_ref[0])[0],
                               [0.002 * np.linalg.norm(C_ref[0]), 0, 0], atol=1e-9)
    C_eq = np.array([5500.0, 0.0, 0.0])
    np.testing.assert_allclose(position_components(C_eq + [0, 0, 2.0], C_eq)[0], [0, 0, 2.0])
    np.testing.assert_allclose(position_components(C_eq + [0, 2.0, 0], C_eq)[0], [0, 2.0, 0])


def test_swing_about_the_body_moves_the_boresight_not_the_landmarks(camera):
    """A 2 km sideways move at 5,500 km, re-pointed at the centre: the weak mode of figure 06."""
    C = np.array([4000.0, -3000.0, 2000.0])
    C *= 5500.0 / np.linalg.norm(C)
    X = rng.normal(size=(500, 3))
    X = 250.0 * X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X[(X @ C) > 0]  # near side
    axis = np.cross(C, rng.normal(size=3))
    C2 = Rotation.from_rotvec(axis / np.linalg.norm(axis) * 2.0 / 5500.0).apply(C)
    R, R2 = look_at(C), look_at(C2)

    split = position_components(C2, C)[0]
    assert abs(split[0]) < 1e-3 and abs(np.hypot(*split[1:]) - 2.0) < 1e-3
    du, dv, _ = pointing_offsets(R2, R, camera)
    boresight = np.hypot(du, dv)
    assert abs(boresight - 2.0 / 5500.0 * camera.fx) < 0.05  # about 4 px
    shift = np.linalg.norm(camera.project((X - C2) @ R2.T) - camera.project((X - C) @ R.T), axis=1)
    assert np.median(shift) < 0.2 and shift.max() < 0.25
    # the same 2 km along the line of sight barely moves the landmarks either
    shift_los = np.linalg.norm(camera.project((X - 1.0004 * C) @ R.T)
                               - camera.project((X - C) @ R.T), axis=1)
    assert shift_los.max() < 0.25
