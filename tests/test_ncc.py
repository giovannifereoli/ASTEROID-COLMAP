import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from asteroid_colmap.ncc import Templates, estimate_pose, local_frames, ncc_maps, ncc_peaks

rng = np.random.default_rng(0)


def look_at(C):
    z = -C / np.linalg.norm(C)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def scene(camera, n=300, range_km=5500.0):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    C = range_km * d
    u = rng.normal(size=(4 * n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[u @ d > 0.3][:n]
    X = 250.0 * u * rng.uniform(0.95, 1.05, size=(len(u), 1))
    R = look_at(C)
    return X, camera.project((X - C) @ R.T), R, C


def test_ncc_maps_peak_at_known_offset():
    regions = rng.normal(size=(3, 40, 40))
    shifts = [(5, 7), (0, 0), (12, 3)]
    templates = np.stack([regions[k, dy:dy + 15, dx:dx + 15] for k, (dy, dx) in enumerate(shifts)])
    maps = ncc_maps(regions, templates)
    assert maps.shape == (3, 26, 26)
    assert np.nanmax(np.abs(maps)) <= 1.0
    for k, (dy, dx) in enumerate(shifts):
        assert maps[k, dy, dx] == pytest.approx(1.0)
        assert np.unravel_index(np.nanargmax(maps[k]), maps[k].shape) == (dy, dx)


def test_ncc_maps_ignores_masked_pixels():
    region = rng.normal(size=(1, 40, 40))
    template = region[:, 10:25, 8:23].copy()
    template[:, :5, :5] = np.nan
    region[:, 10:15, 8:13] = 50.0  # only under the masked pixels
    maps = ncc_maps(region, template)
    assert maps[0, 10, 8] == pytest.approx(1.0)


def test_ncc_maps_is_nan_without_variance():
    maps = ncc_maps(np.ones((1, 20, 20)), rng.normal(size=(1, 5, 5)))
    assert np.isnan(maps).all()


def test_ncc_peaks_subpixel_and_runner_up():
    yy, xx = np.mgrid[:20, :20]
    first = 1.0 - 0.02 * ((yy - 6.3) ** 2 + (xx - 9.7) ** 2)
    second = 0.6 - 0.02 * ((yy - 15) ** 2 + (xx - 3) ** 2)
    border = 1.0 - 0.02 * (yy**2 + (xx - 10) ** 2)
    p = ncc_peaks(np.stack([np.maximum(first, second), border]))
    assert p["dx"][0] == pytest.approx(9.7)
    assert p["dy"][0] == pytest.approx(6.3)
    assert p["ncc"][0] == pytest.approx(first[6, 10])
    assert p["second"][0] == pytest.approx(0.6)
    assert p["interior"].tolist() == [True, False]
    assert np.isnan(p["dx"][1]) and np.isnan(p["dy"][1])


def test_estimate_pose_recovers_attitude(camera):
    X, uv, R, C = scene(camera)
    uv = uv + rng.normal(scale=0.3, size=uv.shape)
    out = rng.choice(len(X), 15, replace=False)
    uv[out] += rng.uniform(10.0, 30.0, size=(15, 2)) * rng.choice([-1, 1], size=(15, 2))
    R0 = Rotation.from_rotvec(np.radians([0.05, -0.1, 0.2])).as_matrix() @ R
    fit = estimate_pose(X, uv, R0, C, camera)
    assert fit.success
    assert not fit.inliers[out].any()
    assert fit.inliers.sum() > 0.95 * (len(X) - 15)
    assert fit.prefit_rms_px > 10.0
    assert fit.postfit_rms_px == pytest.approx(0.3 * np.sqrt(2), rel=0.15)
    err = Rotation.from_matrix(fit.R @ R.T).as_rotvec()
    assert np.all(np.abs(err) * 206264.806 < 4 * fit.sigma_rot_arcsec)
    np.testing.assert_allclose(fit.C, C)


def test_estimate_pose_needs_min_points(camera):
    X, uv, R, C = scene(camera, n=5)
    fit = estimate_pose(X, uv, R, C, camera)
    assert not fit.success
    np.testing.assert_array_equal(fit.R, R)


def test_templates_save_load_round_trip(tmp_path):
    L, K, M = 2, 3, 5
    t = Templates(
        landmark_id=np.array([7, 42]), X=rng.normal(size=(L, 3)), e=rng.normal(size=(L, 3)),
        N=rng.normal(size=(L, 3)), n=rng.normal(size=(L, 3)), height=rng.normal(size=(L, M, M)),
        maplet=rng.normal(size=(L, K, M, M)).astype(np.float16),
        view_image=np.array([["a.png", "b.png", ""], ["c.png", "", ""]]),
        view_sun=rng.normal(size=(L, K, 3)), view_dir=rng.normal(size=(L, K, 3)),
        view_emission_deg=rng.uniform(size=(L, K)), view_incidence_deg=rng.uniform(size=(L, K)),
        view_gsd_km=rng.uniform(size=(L, K)), view_contrast=rng.uniform(size=(L, K)),
        spacing_km=0.492, flip="vertical")
    u = Templates.load(t.save(tmp_path / "catalog" / "templates.npz"))
    assert (len(u), u.size, u.spacing_km, u.flip) == (2, 5, 0.492, "vertical")
    assert u.num_views.tolist() == [2, 1]
    assert u.maplet.dtype == np.float16
    for k in ("landmark_id", "X", "height", "maplet", "view_image", "view_dir", "view_gsd_km"):
        np.testing.assert_array_equal(getattr(u, k), getattr(t, k))


def test_local_frames_follow_a_tilted_plane(body):
    n_true = np.array([np.cos(np.radians(20)), 0.0, np.sin(np.radians(20))])
    a, b = np.array([0.0, 1.0, 0.0]), np.cross(n_true, [0.0, 1.0, 0.0])
    s, t = rng.uniform(-10, 10, size=(2, 400))
    centre = np.array([body.radii_km[0], 0.0, 0.0])
    points = centre + s[:, None] * a + t[:, None] * b + rng.normal(scale=0.01, size=(400, 1)) * n_true
    pole = np.array([0.0, 0.0, body.radii_km[2]])  # no neighbours: ellipsoid normal
    e, N, n = local_frames(np.vstack([centre, pole]), points, body.radii_km)
    np.testing.assert_allclose(n[0], n_true, atol=1e-3)
    np.testing.assert_allclose(n[1], [0.0, 0.0, 1.0], atol=1e-12)
    assert e[0, 2] == pytest.approx(0.0)
    np.testing.assert_allclose(np.cross(e, N), n, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0)
