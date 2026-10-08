import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from asteroid_colmap.geometry import body_from_j2000, rotation_angle_deg
from asteroid_colmap.pose import fit_trajectory, local_axes, posit_pose, solve_pnp, solve_pose

rng = np.random.default_rng(0)


def look_at(C):
    z = -C / np.linalg.norm(C)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def scene(camera, n=400, range_km=5500.0):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    C = range_km * d
    u = rng.normal(size=(4 * n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[u @ d > 0.3][:n]
    X = 250.0 * u * rng.uniform(0.95, 1.05, size=(len(u), 1))
    R = look_at(C)
    return X, camera.project((X - C) @ R.T), R, C


def with_outliers(uv, fraction=0.15):
    uv = uv + rng.normal(scale=0.3, size=uv.shape)
    out = rng.random(len(uv)) < fraction
    uv[out] += rng.uniform(10.0, 50.0, size=(out.sum(), 2)) * rng.choice([-1, 1], size=(out.sum(), 2))
    return uv, out


def test_posit_pose_is_exact_without_noise(camera):
    for _ in range(5):
        X, uv, R, C = scene(camera, n=50)
        R2, C2 = posit_pose(X, camera.pixel_rays(uv[:, 0], uv[:, 1]))
        assert rotation_angle_deg(R2, R) * 3600 < 0.01
        np.testing.assert_allclose(C2, C, atol=1e-4)


def test_solve_pnp_rejects_outliers(camera):
    X, uv, R, C = scene(camera)
    uv, out = with_outliers(uv)
    R2, C2, inl = solve_pnp(X, uv, camera)
    assert not inl[out].any()
    assert inl[~out].mean() > 0.95
    assert rotation_angle_deg(R2, R) * 3600 < 150  # 1-sigma is ~35" per axis here
    assert np.linalg.norm(C2 - C) < 5.0


def test_solve_pnp_needs_four_points(camera):
    X, uv, _, _ = scene(camera, n=3)
    R, C, inl = solve_pnp(X, uv, camera)
    assert R is None and C is None
    assert inl.tolist() == [False] * 3


def test_solve_pose_with_outliers(camera):
    X, uv, R, C = scene(camera)
    uv, out = with_outliers(uv)
    uv[0] = np.nan
    rp = solve_pose(X, uv, camera)
    assert rp.success
    assert not rp.inliers[out].any() and not rp.inliers[0]
    assert rp.inliers[~out].mean() > 0.95
    assert rp.rms_px == pytest.approx(0.3 * np.sqrt(2), rel=0.15)
    assert rp.range_km == pytest.approx(np.linalg.norm(C), abs=1.0)
    uv_c, sigma_c = rp.target_px(camera)
    assert np.all(np.abs(uv_c - camera.project(R @ -C)[0]) < 4 * sigma_c)
    assert np.all(sigma_c < 0.2)


def test_solve_pose_covariance_matches_scatter(camera):
    X, uv, R, C = scene(camera)
    E = local_axes(C)
    z = []
    for _ in range(30):
        rp = solve_pose(X, uv + rng.normal(scale=0.3, size=uv.shape), camera)
        rot = Rotation.from_matrix(rp.R @ R.T).as_rotvec() * 206264.806
        z.append(np.r_[rot / rp.attitude_sigma_arcsec(), E @ (rp.C - C) / rp.position_sigma_km()])
    z = np.array(z)
    assert 0.7 < np.sqrt(np.mean(z**2)) < 1.4
    sig = rp.position_sigma_km()
    assert sig[0] < sig[1] and sig[0] < sig[2]  # the range is better determined than the side


def test_solve_pose_fails_with_too_few_points(camera):
    X, uv, _, _ = scene(camera, n=8)
    rp = solve_pose(X, uv, camera)
    assert not rp.success
    assert np.isnan(rp.R).all() and np.isnan(rp.C).all()


def test_local_axes_are_orthonormal():
    C = np.array([1102.5, -2285.0, -4857.1])
    E = local_axes(C)
    np.testing.assert_allclose(E @ E.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(E[0], C / np.linalg.norm(C))
    assert E[1, 2] == pytest.approx(0.0)
    assert E[2, 2] > 0


def trajectory_rows(body, sequence, n, sigma_km=(0.1, 0.4, 0.4)):
    t0 = dt.datetime(2011, 7, 24, 6, 0, 0)
    rows = []
    for k in range(n):
        h = 5.0 * k / max(n - 1, 1)
        t = t0 + dt.timedelta(hours=h)
        y = np.array([3000.0, -2000.0, 4000.0]) + np.array([10.0, 5.0, -3.0]) * h \
            + 0.5 * np.array([0.5, -0.2, 0.1]) * h**2
        Rb = body_from_j2000(body, t)
        C = Rb @ y
        E = local_axes(C)
        P = E.T @ np.diag(np.square(sigma_km)) @ E  # body-fixed
        noisy = y + Rb.T @ E.T @ (np.asarray(sigma_km) * rng.normal(size=3))
        rows.append({
            "image": f"{sequence}_{k:03d}.png", "sequence": sequence, "success": True,
            "utc": t.isoformat(timespec="milliseconds"),
            "j2000_x_km": noisy[0], "j2000_y_km": noisy[1], "j2000_z_km": noisy[2],
            "cov_xx_km2": P[0, 0], "cov_xy_km2": P[0, 1], "cov_xz_km2": P[0, 2],
            "cov_yy_km2": P[1, 1], "cov_yz_km2": P[1, 2], "cov_zz_km2": P[2, 2],
            "label_x_km": C[0], "label_y_km": C[1], "label_z_km": C[2]})
    return rows


def test_fit_trajectory_chi2_and_smoothing(body):
    rel = pd.DataFrame(trajectory_rows(body, "A", 200) + trajectory_rows(body, "B", 2))
    traj = fit_trajectory(rel, body, degree=2)
    assert set(traj.sequence) == {"A"} and len(traj) == 200
    assert (traj.degree == 2).all()
    assert traj.chi2_dof.iloc[0] == pytest.approx(1.0, abs=0.25)
    err = traj[["range_minus_label_km", "east_minus_label_km", "north_minus_label_km"]].to_numpy()
    sig = traj[["sigma_range_km", "sigma_east_km", "sigma_north_km"]].to_numpy()
    assert np.sqrt(np.mean((err / sig) ** 2)) < 2.0
    assert np.sqrt(np.mean(err[:, 1] ** 2)) < 0.4 / 3  # smoothing beats a single image
    resid = traj[["residual_range_km", "residual_east_km", "residual_north_km"]].to_numpy()
    assert np.std(resid, axis=0) == pytest.approx([0.1, 0.4, 0.4], rel=0.2)


def test_fit_trajectory_skips_short_sequences(body):
    rel = pd.DataFrame(trajectory_rows(body, "B", 2))
    assert fit_trajectory(rel, body).empty
    short = fit_trajectory(pd.DataFrame(trajectory_rows(body, "C", 3)), body, degree=2)
    assert (short.degree == 1).all()
