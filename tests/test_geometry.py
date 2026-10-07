import datetime as dt

import numpy as np
import pytest

from asteroid_colmap.geometry import (cart_to_latlonr, ellipsoid_radius, latlon_to_unit,
                                      matrix_to_quat, quat_to_matrix, ray_ellipsoid,
                                      rotation_angle_deg, tdb_minus_utc, umeyama)
from asteroid_colmap.georef import robust_similarity
from asteroid_colmap.metadata import label_geometry, label_record

rng = np.random.default_rng(0)


def random_rotation():
    q = rng.normal(size=4)
    return quat_to_matrix(q / np.linalg.norm(q))


def test_quat_to_matrix_is_proper_rotation():
    R = quat_to_matrix([0.3, -0.2, 0.9, 0.1])
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_quat_matrix_round_trip():
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q2 = matrix_to_quat(quat_to_matrix(q))
        assert q2[0] >= 0
        np.testing.assert_allclose(q2, q * np.sign(q[0]), atol=1e-10)


def test_matrix_to_quat_near_180_degrees():
    R = quat_to_matrix([1e-9, 0.0, 0.6, 0.8])
    np.testing.assert_allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-10)


def test_rotation_angle():
    a = np.radians(37.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert rotation_angle_deg(np.eye(3), Rz) == pytest.approx(37.0)


def test_umeyama_recovers_similarity():
    src = rng.normal(size=(50, 3))
    R, s, t = random_rotation(), 3.7, np.array([10.0, -4.0, 2.0])
    s2, R2, t2 = umeyama(src, s * src @ R.T + t)
    assert s2 == pytest.approx(s)
    np.testing.assert_allclose(R2, R, atol=1e-10)
    np.testing.assert_allclose(t2, t, atol=1e-10)


def test_umeyama_never_returns_a_reflection():
    src = rng.normal(size=(30, 3))
    _, R, _ = umeyama(src, src * np.array([1.0, 1.0, -1.0]))
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_robust_similarity_rejects_outliers():
    src = rng.normal(size=(40, 3)) * 100
    R, s, t = random_rotation(), 2.0, np.array([5.0, 0.0, 0.0])
    dst = s * src @ R.T + t + rng.normal(scale=0.01, size=src.shape)
    dst[:3] += 500.0
    sim, res, keep = robust_similarity(src, dst)
    assert not keep[:3].any() and keep[3:].all()
    assert sim.scale == pytest.approx(s, rel=1e-4)


def test_tdb_minus_utc():
    assert tdb_minus_utc(dt.datetime(2011, 7, 24)) == pytest.approx(66.184)
    assert tdb_minus_utc(dt.datetime(2020, 1, 1)) == pytest.approx(69.184)
    with pytest.raises(ValueError):
        tdb_minus_utc(dt.datetime(1990, 1, 1))


def test_latlon_round_trip():
    lat, lon = np.array([-80.0, 0.0, 45.0]), np.array([10.0, 200.0, 359.0])
    lat2, lon2, r = cart_to_latlonr(3.0 * latlon_to_unit(lat, lon))
    np.testing.assert_allclose(lat2, lat)
    np.testing.assert_allclose(lon2, lon)
    np.testing.assert_allclose(r, 3.0)


def test_ellipsoid_radius_axes():
    radii = (286.3, 278.6, 223.2)
    np.testing.assert_allclose(ellipsoid_radius(np.array([0, 0, 90]), np.array([0, 90, 0]), radii),
                               radii)


def test_ray_ellipsoid_hits_near_side_and_misses():
    radii = (3.0, 2.0, 1.0)
    origin = np.array([10.0, 0.0, 0.0])
    dirs = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.5, 0.0]])
    pts, hit = ray_ellipsoid(origin, dirs, radii)
    assert hit.tolist() == [True, False, False]
    np.testing.assert_allclose(pts[0], [3.0, 0.0, 0.0])
    assert np.isnan(pts[1]).all()


def test_rotation_model_reproduces_label_sub_spacecraft_point(label_path, body):
    """W0 = 285.39 (IAU 2015, Claudia double-prime) maps the J2000 label vector onto the
    label sub-spacecraft point; the older Claudia W0 = 75.39 would be 210 deg off."""
    rec = label_record(label_path, body)
    assert rec["subsc_lat_model_deg"] == pytest.approx(15.1976, abs=1e-3)
    assert rec["subsc_lon_model_deg"] == pytest.approx(301.656, abs=1e-3)
    assert rec["rotation_model_check_deg"] < 1e-3
    assert rec["sequence"] == "RC3"
    assert rec["filter"] == 1
    assert rec["image"] == "FC21B0003112_11205060002F1C.png"


def test_label_geometry_boresight_points_at_vesta(label_path, body):
    rec = label_record(label_path, body)
    g = label_geometry(rec, body)
    to_centre = -g["sc_bf"] / np.linalg.norm(g["sc_bf"])
    boresight = g["R_cam_from_bf"].T @ np.array([0.0, 0.0, 1.0])
    # Vesta is ~3 deg across at this range and was imaged near the frame centre
    assert np.degrees(np.arccos(boresight @ to_centre)) < 0.3
    np.testing.assert_allclose(np.linalg.norm(g["sc_bf"]), 5479.761, rtol=1e-6)
