"""Relative pose of the camera with respect to the body from landmark measurements.

The NCC matches of :mod:`~asteroid_colmap.ncc` give, for every new image, pixel measurements of
catalog landmarks whose body-fixed positions are known: 2-D/3-D correspondences. This module
solves the full camera pose from them, i.e. the spacecraft position relative to the body
centre and the camera attitude (6 degrees of freedom), without using the pose in the label:

1. :func:`posit_pose`: scaled orthographic projection with an iterative perspective
   correction (POSIT), which needs no starting pose; run inside RANSAC by :func:`solve_pnp` to
   reject wrong matches;
2. :func:`solve_pose`: robust least squares on the pixel residuals, started from the RANSAC
   solution, with the 6x6 covariance (:func:`~asteroid_colmap.ncc.estimate_pose` without a
   position prior);
3. :func:`relative_poses`: every image of a matching run, compared with the label pose, the
   NCC attitude-only fit and, for catalog images, the georeferenced COLMAP pose;
4. :func:`fit_trajectory`: a weighted polynomial through each sequence's positions in J2000,
   which smooths the per-image noise and checks the formal uncertainties.

The matching itself searches around the label prediction, so it needs an a priori pose; the
pose solution does not. From thousands of kilometres with a 5.5 deg field of view, the range
is set by the apparent size of the landmark pattern and is the best-determined direction. The
weak combination is a sideways displacement compensated by a rotation that keeps the body
centre on the same pixel; only the depth relief of the visible surface separates the two.
Position uncertainties are therefore reported along the range (radial) and across it (east and
north at the sub-spacecraft point), next to the pixel position of the body centre, which is
well determined.

All poses map body-fixed vectors into the camera frame (x right, y down, z along the
boresight); positions are body-fixed in km; pixel coordinates follow COLMAP.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .camera import FramingCamera
from .config import Body
from .geometry import (body_from_j2000, cart_to_latlonr, local_axes, matrix_to_quat, quat_to_matrix,
                       rotation_angle_deg)
from .metadata import label_geometry
from .ncc import _ARCSEC, _rms, estimate_pose

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- solvers

def posit_pose(X: np.ndarray, rays: np.ndarray, iterations: int = 30, tol: float = 1e-12):
    """Camera pose from four or more 2-D/3-D correspondences by POSIT.

    POSIT (DeMenthon and Davis, 1995) writes the perspective projection as a scaled
    orthographic one corrected by the relative depth of every point,
    ``x_i (1 + eps_i) = (r1 . (X_i - X0)) / Z0 + x0``, solves the linear part by least squares
    (eight unknowns), makes the first two rows of the rotation orthonormal, takes the third
    as their cross product and updates ``eps_i = r3 . (X_i - X0) / Z0`` until it stops
    changing. Unlike the 11-parameter projective DLT it knows the camera is calibrated, so it
    stays well conditioned in a narrow field of view, where the depth row of a projective
    camera is barely observable; the convergence is fastest there (``eps`` is a few percent).
    No a priori pose is needed; the result is the start of :func:`solve_pose`.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, 3)`` body-fixed points (km), ``n >= 4``, not all on a plane.
    rays : numpy.ndarray
        ``(n, 3)`` camera-frame directions of their measurements
        (:meth:`~asteroid_colmap.camera.FramingCamera.pixel_rays`).
    iterations : int
        Most depth-correction iterations.
    tol : float
        Convergence threshold on ``eps``.

    Returns
    -------
    R : numpy.ndarray
        ``(3, 3)`` body-fixed -> camera rotation.
    C : numpy.ndarray
        ``(3,)`` camera position (km).
    """
    X, rays = np.asarray(X, float), np.asarray(rays, float)
    x = rays[:, :2] / rays[:, 2:3]
    X0 = X.mean(axis=0)
    D = X - X0
    A = np.column_stack([D, np.ones(len(D))])
    eps = np.zeros(len(X))
    for _ in range(iterations):
        sol = np.linalg.lstsq(A, x * (1.0 + eps)[:, None], rcond=None)[0]  # (4, 2)
        U, S, Vt = np.linalg.svd(sol[:3].T, full_matrices=False)  # rows: r1 / Z0, r2 / Z0
        r12 = U @ Vt
        R = np.vstack([r12, np.cross(r12[0], r12[1])])
        Z0 = 1.0 / S.mean()
        new = D @ R[2] / Z0
        done = np.max(np.abs(new - eps)) < tol
        eps = new
        if done:
            break
    t = Z0 * np.r_[sol[3], 1.0] - R @ X0  # X0 lies at Z0 * (x0, y0, 1) in the camera frame
    return R, -R.T @ t


def _reprojection(X, uv, R, C, camera):
    """Pixel residual norm of every point; infinite for points behind the camera."""
    Xc = (X - C) @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(camera.project(Xc) - uv, axis=1)
    return np.where(Xc[:, 2] > 0, r, np.inf)


def solve_pnp(X: np.ndarray, uv: np.ndarray, camera: FramingCamera, threshold_px: float = 2.0,
              sample: int = 12, max_iterations: int = 500, confidence: float = 0.999,
              seed: int = 0):
    """Robust pose from scratch: :func:`posit_pose` inside RANSAC.

    Each iteration solves POSIT on ``sample`` random correspondences (more than the minimal
    four, which averages the measurement noise) and counts the points that reproject within
    ``threshold_px``. Every new best hypothesis is re-solved on its own inliers until their
    number stops growing (local optimisation), and the iteration count adapts to the best
    inlier fraction so far.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, 3)`` body-fixed points (km).
    uv : numpy.ndarray
        ``(n, 2)`` measured pixel positions (finite).
    camera : FramingCamera
        Camera model.
    threshold_px : float
        Inlier threshold.
    sample : int
        Correspondences per hypothesis (at least 4).
    max_iterations : int
        Upper bound on the hypotheses.
    confidence : float
        Probability of drawing at least one all-inlier sample.
    seed : int
        Random seed (the result is reproducible).

    Returns
    -------
    R, C : numpy.ndarray
        Pose (body-fixed -> camera, position in km); ``None`` when no hypothesis has
        ``sample`` inliers.
    inliers : numpy.ndarray
        Boolean mask.
    """
    X, uv = np.asarray(X, float), np.asarray(uv, float)
    n = len(X)
    sample = max(4, min(sample, n))
    if n < 4:
        return None, None, np.zeros(n, bool)
    rays = camera.pixel_rays(uv[:, 0], uv[:, 1])
    rng = np.random.default_rng(seed)
    best, pose, need, it = np.zeros(n, bool), (None, None), max_iterations, 0
    while it < min(need, max_iterations):
        it += 1
        s = rng.choice(n, sample, replace=False)
        R, C = posit_pose(X[s], rays[s])
        inl = _reprojection(X, uv, R, C, camera) < threshold_px
        if inl.sum() <= max(best.sum(), sample - 1):
            continue
        while True:  # local optimisation: re-solve on the consensus set while it holds
            R2, C2 = posit_pose(X[inl], rays[inl])
            inl2 = _reprojection(X, uv, R2, C2, camera) < threshold_px
            if inl2.sum() < inl.sum():
                break
            grew = inl2.sum() > inl.sum()
            R, C, inl = R2, C2, inl2
            if not grew:
                break
        best, pose = inl, (R, C)
        w = inl.mean() ** sample  # chance that a sample is all inliers
        need = 0 if w >= 1 else (max_iterations if w < 1e-12
                                 else np.log(1 - confidence) / np.log1p(-w))
    return pose[0], pose[1], best


@dataclass
class PoseOptions:
    """Settings of :func:`solve_pose`.

    Attributes
    ----------
    sigma_px : float
        Measurement noise (px), for the weights and the covariance.
    ransac_threshold_px : float
        RANSAC inlier threshold (px).
    ransac_sample : int
        Correspondences per RANSAC hypothesis.
    ransac_iterations : int
        Upper bound on the RANSAC hypotheses.
    k_sigma, floor_px : float
        Outlier threshold of the refinement:
        ``max(median + k_sigma * 1.4826 MAD, floor_px)``.
    min_points : int
        Fewest inliers for a pose.
    rounds : int
        Refinement / re-selection rounds after RANSAC.
    seed : int
        RANSAC random seed.
    """
    sigma_px: float = 0.3
    ransac_threshold_px: float = 2.0
    ransac_sample: int = 12
    ransac_iterations: int = 500
    k_sigma: float = 3.0
    floor_px: float = 1.0
    min_points: int = 12
    rounds: int = 3
    seed: int = 0


@dataclass
class RelativePose:
    """Camera pose relative to the body, with its uncertainty (:func:`solve_pose`).

    Attributes
    ----------
    R : numpy.ndarray
        ``(3, 3)`` body-fixed -> camera rotation.
    C : numpy.ndarray
        ``(3,)`` camera position, body-fixed (km).
    covariance : numpy.ndarray
        ``(6, 6)`` covariance of a small rotation of the camera (rotation vector in the camera
        frame, rad) and of the position (km).
    inliers : numpy.ndarray
        Boolean mask of the correspondences kept.
    rms_px : float
        RMS pixel residual of the inliers.
    ransac_inliers : int
        Inliers of the RANSAC solution.
    success : bool
        False when too few correspondences were consistent; the pose is then NaN.
    """
    R: np.ndarray
    C: np.ndarray
    covariance: np.ndarray
    inliers: np.ndarray
    rms_px: float
    ransac_inliers: int
    success: bool = True

    @property
    def range_km(self) -> float:
        """Distance from the body centre (km)."""
        return float(np.linalg.norm(self.C))

    @property
    def target_cam(self) -> np.ndarray:
        """``(3,)`` body centre in the camera frame (km)."""
        return self.R @ -self.C

    @property
    def subsc_lat_lon(self) -> tuple[float, float]:
        """Planetocentric latitude and east longitude of the sub-spacecraft point (deg)."""
        lat, lon, _ = cart_to_latlonr(self.C)
        return float(lat[0]), float(lon[0])

    def local_axes(self) -> np.ndarray:
        """``(3, 3)`` rows: radial (up), east and north unit vectors at the sub-spacecraft point."""
        return local_axes(self.C)

    def position_sigma_km(self) -> np.ndarray:
        """1-sigma of the position along the radial (range), east and north axes (km)."""
        E = self.local_axes()
        return np.sqrt(np.diag(E @ self.covariance[3:, 3:] @ E.T))

    def attitude_sigma_arcsec(self) -> np.ndarray:
        """1-sigma of the attitude about the camera x, y and z axes (arcsec)."""
        return np.sqrt(np.diag(self.covariance[:3, :3])) * _ARCSEC

    def target_px(self, camera: FramingCamera):
        """Pixel position of the body centre and its 1-sigma.

        Parameters
        ----------
        camera : FramingCamera
            Camera model.

        Returns
        -------
        uv : numpy.ndarray
            ``(2,)`` body-centre pixel.
        sigma : numpy.ndarray
            ``(2,)`` its 1-sigma (px), from the pose covariance.
        """
        def centre(d):
            R = Rotation.from_rotvec(d[:3]).as_matrix() @ self.R
            return camera.project(R @ -(self.C + d[3:]))[0]

        uv = centre(np.zeros(6))
        steps = np.r_[np.full(3, 1e-7), np.full(3, 1e-3)]
        J = np.column_stack([(centre(np.eye(6)[k] * h) - uv) / h for k, h in enumerate(steps)])
        return uv, np.sqrt(np.diag(J @ self.covariance @ J.T))


def solve_pose(X: np.ndarray, uv: np.ndarray, camera: FramingCamera,
               opt: PoseOptions | None = None) -> RelativePose:
    """Full 6-DOF camera pose from landmark measurements, without an a priori pose.

    :func:`solve_pnp` gives a first pose and rejects gross outliers; the robust least-squares
    fit of :func:`~asteroid_colmap.ncc.estimate_pose` (attitude and position, no prior) then
    refines it and returns the covariance, scaled by the a posteriori variance factor when it
    exceeds one. After each fit the correspondences within ``ransac_threshold_px`` of the
    refined pose are selected again from all of them, so points that the RANSAC pose
    misjudged are recovered, and ``estimate_pose`` rejects the remaining outliers.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, 3)`` landmark positions, body-fixed (km).
    uv : numpy.ndarray
        ``(n, 2)`` measured pixel positions; rows with NaN are ignored.
    camera : FramingCamera
        Camera model.
    opt : PoseOptions, optional
        Settings.

    Returns
    -------
    RelativePose
    """
    opt = opt or PoseOptions()
    X, uv = np.asarray(X, float), np.asarray(uv, float)
    ok = np.isfinite(uv).all(axis=1) & np.isfinite(X).all(axis=1)
    inliers = np.zeros(len(X), bool)
    fail = RelativePose(np.full((3, 3), np.nan), np.full(3, np.nan), np.full((6, 6), np.nan),
                        inliers, float("nan"), 0, success=False)
    if ok.sum() < opt.min_points:
        return fail
    idx = np.flatnonzero(ok)
    R0, C0, inl = solve_pnp(X[idx], uv[idx], camera, opt.ransac_threshold_px, opt.ransac_sample,
                            opt.ransac_iterations, seed=opt.seed)
    if R0 is None or inl.sum() < opt.min_points:
        fail.ransac_inliers = int(inl.sum())
        return fail
    n_ransac = int(inl.sum())
    sel = idx[inl]
    for k in range(opt.rounds):
        fit = estimate_pose(X[sel], uv[sel], R0, C0, camera, sigma_px=opt.sigma_px,
                            estimate_position=True, position_sigma_km=None, k_sigma=opt.k_sigma,
                            floor_px=opt.floor_px, min_points=opt.min_points)
        kept = sel[fit.inliers]
        if not fit.success or k == opt.rounds - 1:
            break
        R0, C0 = fit.R, fit.C
        new = idx[_reprojection(X[idx], uv[idx], R0, C0, camera) < opt.ransac_threshold_px]
        if np.array_equal(new, sel):
            break
        sel = new
    inliers[kept] = True
    if not fit.success:
        fail.inliers, fail.ransac_inliers = inliers, n_ransac
        return fail
    return RelativePose(fit.R, fit.C, fit.covariance, inliers, fit.postfit_rms_px, n_ransac)


# --------------------------------------------------------------------------- batch

def _split(d: np.ndarray, axes: np.ndarray) -> dict:
    """Range, east and north components of a position difference."""
    r, e, n = axes @ d
    return {"range": r, "east": e, "north": n, "position": float(np.linalg.norm(d))}


def _sfm_poses(cams: pd.DataFrame | None) -> dict:
    """``image -> (R, C)`` of the registered images of a catalog ``cameras.csv``."""
    if cams is None or "sfm_qw" not in cams.columns:
        return {}
    reg = cams[cams.registered.astype(bool)]
    return {r.image: (quat_to_matrix([r.sfm_qw, r.sfm_qx, r.sfm_qy, r.sfm_qz]),
                      np.array([r.sfm_x_km, r.sfm_y_km, r.sfm_z_km]))
            for r in reg.itertuples(index=False)}


def relative_poses(matches: pd.DataFrame, meta: pd.DataFrame, body: Body, camera: FramingCamera,
                   opt: PoseOptions | None = None, ncc_poses: pd.DataFrame | None = None,
                   catalog_cameras: pd.DataFrame | None = None,
                   progress: bool = True) -> pd.DataFrame:
    """Solve the relative pose of every image of a matching run and compare it.

    The accepted NCC matches of each image (``accepted``: correlation and margin tests, before
    the attitude-only fit) are the correspondences; :func:`solve_pose` uses neither the label
    pose nor the attitude-only fit.

    Parameters
    ----------
    matches : pandas.DataFrame
        Output of :func:`~asteroid_colmap.ncc.match_images` (``image``, ``x_km..z_km``, ``u``,
        ``v``, ``accepted``).
    meta : pandas.DataFrame
        Metadata of the same images (label geometry, ``utc``, ``sequence``).
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    opt : PoseOptions, optional
        Settings.
    ncc_poses : pandas.DataFrame, optional
        ``poses.csv`` of the run, for the comparison with the attitude-only fit.
    catalog_cameras : pandas.DataFrame, optional
        The catalog's ``cameras.csv``, for the comparison with the COLMAP poses of catalog
        images.
    progress : bool
        Show a progress bar.

    Returns
    -------
    pandas.DataFrame
        One row per image:

        * ``num_points``, ``ransac_inliers``, ``num_inliers``, ``rms_px``, ``success``;
        * the pose: ``x_km..z_km`` (body-fixed), ``range_km``, ``subsc_lat_deg``,
          ``subsc_lon_deg``, ``qw..qz`` (body-fixed -> camera), ``target_x_km..target_z_km``
          (body centre in the camera frame), ``target_u_px``, ``target_v_px`` (its pixel),
          ``j2000_x_km..j2000_z_km`` (position in J2000) and ``j2000_qw..j2000_qz``
          (J2000 -> camera, comparable with the label quaternion);
        * 1-sigma: ``sigma_range_km``, ``sigma_east_km``, ``sigma_north_km``,
          ``sigma_rx_arcsec..sigma_rz_arcsec``, ``sigma_target_u_px``, ``sigma_target_v_px``
          and the body-fixed position covariance ``cov_xx_km2 .. cov_zz_km2``;
        * estimate minus label: ``range_minus_label_km``, ``east_minus_label_km``,
          ``north_minus_label_km``, ``position_minus_label_km``, ``attitude_minus_label_arcsec``,
          ``target_du_px``, ``target_dv_px``; ``label_rms_px`` is the label pose's RMS on the
          same inliers;
        * with ``ncc_poses``: ``attitude_minus_ncc_arcsec``, ``ncc_rms_px`` and the
          attitude-only fit minus the label, ``ncc_attitude_minus_label_arcsec``;
        * with ``catalog_cameras`` and a registered image: the same differences against the
          COLMAP pose (``*_minus_sfm_*``), ``sfm_rms_px`` and
          ``sfm_attitude_minus_label_arcsec``.
    """
    opt = opt or PoseOptions()
    look = meta.set_index("image")
    sfm = _sfm_poses(catalog_cameras)
    ncc = ncc_poses.set_index("image") if ncc_poses is not None and len(ncc_poses) else None
    rows = []
    groups = matches[matches.accepted.astype(bool)].groupby("image", sort=False)
    for name, m in tqdm(groups, total=groups.ngroups, desc="pose", unit="image",
                        disable=not progress):
        row = look.loc[name]
        X = m[["x_km", "y_km", "z_km"]].to_numpy()
        uv = m[["u", "v"]].to_numpy()
        rp = solve_pose(X, uv, camera, opt)
        g = label_geometry(row, body)
        rec = {"image": name, "sequence": row.get("sequence"), "utc": row.get("utc"),
               "num_points": len(m), "ransac_inliers": rp.ransac_inliers,
               "num_inliers": int(rp.inliers.sum()), "rms_px": rp.rms_px, "success": rp.success}
        if rp.success:
            rec.update(_pose_columns(rp, g, row, body, camera, X[rp.inliers], uv[rp.inliers]))
            if ncc is not None and name in ncc.index:
                q = ncc.loc[name]
                R_n = quat_to_matrix([q.qw, q.qx, q.qy, q.qz])
                C_n = np.array([q.x_km, q.y_km, q.z_km])
                rec["attitude_minus_ncc_arcsec"] = rotation_angle_deg(rp.R, R_n) * 3600
                rec["ncc_attitude_minus_label_arcsec"] = rotation_angle_deg(
                    R_n, g["R_cam_from_bf"]) * 3600
                rec["ncc_rms_px"] = _rms(camera.project((X[rp.inliers] - C_n) @ R_n.T)
                                         - uv[rp.inliers])
            if name in sfm:
                rec.update(_compare(rp, *sfm[name], g["sc_bf"], camera, X[rp.inliers],
                                    uv[rp.inliers], "sfm"))
                rec["sfm_attitude_minus_label_arcsec"] = rotation_angle_deg(
                    sfm[name][0], g["R_cam_from_bf"]) * 3600
            log.info("%s: %d/%d inliers, RMS %.2f px, range %.2f km (+-%.2f), "
                     "minus label: range %+.2f, east %+.2f, north %+.2f km",
                     name, rec["num_inliers"], rec["num_points"], rp.rms_px, rp.range_km,
                     rec["sigma_range_km"], rec["range_minus_label_km"],
                     rec["east_minus_label_km"], rec["north_minus_label_km"])
        else:
            log.warning("%s: no pose from %d matches", name, len(m))
        rows.append(rec)
    return pd.DataFrame(rows)


def _compare(rp: RelativePose, R_ref, C_ref, C_label, camera, X, uv, tag: str) -> dict:
    """Estimate minus a reference pose, split on the axes at the label position."""
    d = _split(rp.C - C_ref, local_axes(C_label))
    out = {f"{k}_minus_{tag}_km": v for k, v in d.items()}
    out[f"attitude_minus_{tag}_arcsec"] = rotation_angle_deg(rp.R, R_ref) * 3600
    t_est = camera.project(rp.target_cam)[0]
    t_ref = camera.project(R_ref @ -C_ref)[0]
    out[f"target_du_{tag}_px" if tag != "label" else "target_du_px"] = t_est[0] - t_ref[0]
    out[f"target_dv_{tag}_px" if tag != "label" else "target_dv_px"] = t_est[1] - t_ref[1]
    out[f"{tag}_rms_px"] = _rms(camera.project((X - C_ref) @ R_ref.T) - uv)
    return out


def _pose_columns(rp: RelativePose, g: dict, row, body: Body, camera: FramingCamera,
                  X: np.ndarray, uv: np.ndarray) -> dict:
    """Pose, uncertainty and label-comparison columns of one :func:`relative_poses` row."""
    q = matrix_to_quat(rp.R)
    lat, lon = rp.subsc_lat_lon
    tc = rp.target_cam
    t_uv, t_sig = rp.target_px(camera)
    R_bf = body_from_j2000(body, dt.datetime.fromisoformat(str(row["utc"])))
    Cj = R_bf.T @ rp.C
    qj = matrix_to_quat(rp.R @ R_bf)
    s_pos, s_att = rp.position_sigma_km(), rp.attitude_sigma_arcsec()
    P = rp.covariance[3:, 3:]
    rec = {
        "x_km": rp.C[0], "y_km": rp.C[1], "z_km": rp.C[2], "range_km": rp.range_km,
        "subsc_lat_deg": lat, "subsc_lon_deg": lon,
        "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
        "target_x_km": tc[0], "target_y_km": tc[1], "target_z_km": tc[2],
        "target_u_px": t_uv[0], "target_v_px": t_uv[1],
        "j2000_x_km": Cj[0], "j2000_y_km": Cj[1], "j2000_z_km": Cj[2],
        "j2000_qw": qj[0], "j2000_qx": qj[1], "j2000_qy": qj[2], "j2000_qz": qj[3],
        "sigma_range_km": s_pos[0], "sigma_east_km": s_pos[1], "sigma_north_km": s_pos[2],
        "sigma_rx_arcsec": s_att[0], "sigma_ry_arcsec": s_att[1], "sigma_rz_arcsec": s_att[2],
        "sigma_target_u_px": t_sig[0], "sigma_target_v_px": t_sig[1],
        "cov_xx_km2": P[0, 0], "cov_xy_km2": P[0, 1], "cov_xz_km2": P[0, 2],
        "cov_yy_km2": P[1, 1], "cov_yz_km2": P[1, 2], "cov_zz_km2": P[2, 2],
        "label_x_km": g["sc_bf"][0], "label_y_km": g["sc_bf"][1], "label_z_km": g["sc_bf"][2],
        "label_range_km": float(np.linalg.norm(g["sc_bf"])),
    }
    rec.update(_compare(rp, g["R_cam_from_bf"], g["sc_bf"], g["sc_bf"], camera, X, uv, "label"))
    return rec


# --------------------------------------------------------------------------- trajectory

def _covariances(rel: pd.DataFrame) -> np.ndarray:
    """``(n, 3, 3)`` body-fixed position covariances from the ``cov_*_km2`` columns."""
    c = {k: rel[f"cov_{k}_km2"].to_numpy() for k in ("xx", "xy", "xz", "yy", "yz", "zz")}
    return np.stack([np.stack([c["xx"], c["xy"], c["xz"]], -1),
                     np.stack([c["xy"], c["yy"], c["yz"]], -1),
                     np.stack([c["xz"], c["yz"], c["zz"]], -1)], -2)


def fit_trajectory(rel: pd.DataFrame, body: Body, degree: int = 2) -> pd.DataFrame:
    """Smooth the per-image positions of each sequence with a polynomial in time.

    Relative to the body centre, the spacecraft moves smoothly in an inertial frame, so the
    positions are rotated to J2000 and each sequence gets a polynomial of ``degree`` per axis,
    fitted by generalised least squares with the full 3x3 covariance of every image. The
    scatter about the fit, divided by the formal 1-sigma, checks the covariance (about 1 when
    it is realistic); the smoothed positions average the per-image noise.

    Parameters
    ----------
    rel : pandas.DataFrame
        Output of :func:`relative_poses`; only rows with ``success`` are used.
    body : Body
        Target body (rotation model).
    degree : int
        Polynomial degree, lowered to ``n - 2`` (at least 1) for a sequence of ``n`` images;
        sequences with fewer than three images are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per image: ``image``, ``sequence``, ``utc``, ``hours`` (since the sequence
        start), ``degree``, the smoothed position ``x_km..z_km`` (body-fixed) and its 1-sigma
        ``sigma_range_km``, ``sigma_east_km``, ``sigma_north_km``, the per-image residual about
        the fit ``residual_range_km``, ``residual_east_km``, ``residual_north_km``, ``chi2_dof``
        of the sequence's fit, and the smoothed position minus the label position
        (``range_minus_label_km``, ``east_minus_label_km``, ``north_minus_label_km``).
        Empty when no sequence has three or more images.
    """
    ok = rel[rel.success.astype(bool)] if len(rel) else rel
    out = []
    for seq, g in ok.groupby("sequence", sort=False):
        n = len(g)
        if n < 3:  # a line through two positions has no redundancy
            continue
        deg = max(1, min(degree, n - 2))
        times = pd.to_datetime(g.utc)
        hours = ((times - times.min()).dt.total_seconds() / 3600.0).to_numpy()
        tc = hours - hours.mean()
        Rb = np.stack([body_from_j2000(body, t.to_pydatetime()) for t in times])
        y = g[["j2000_x_km", "j2000_y_km", "j2000_z_km"]].to_numpy()
        P = np.einsum("nji,njk,nkl->nil", Rb, _covariances(g), Rb)  # to J2000
        W = np.linalg.inv(P)
        V = np.vander(tc, deg + 1, increasing=True)  # (n, deg+1)
        A = np.einsum("np,ij->nipj", V, np.eye(3)).reshape(n, 3, 3 * (deg + 1))
        N = np.einsum("nia,nij,njb->ab", A, W, A)
        b = np.einsum("nia,nij,nj->a", A, W, y)
        Ncov = np.linalg.inv(N)
        coef = Ncov @ b
        yhat = A @ coef
        res = y - yhat
        chi2 = float(np.einsum("ni,nij,nj->", res, W, res))
        dof = max(3 * n - 3 * (deg + 1), 1)
        Phat = np.einsum("nia,ab,njb->nij", A, Ncov, A)
        for k, (_, r) in enumerate(g.iterrows()):
            C_hat = Rb[k] @ yhat[k]
            C_lab = r[["label_x_km", "label_y_km", "label_z_km"]].to_numpy(float)
            E = local_axes(C_lab)
            Pk = E @ Rb[k] @ Phat[k] @ Rb[k].T @ E.T
            rr = E @ Rb[k] @ res[k]
            dl = E @ (C_hat - C_lab)
            out.append({
                "image": r.image, "sequence": seq, "utc": r.utc, "hours": hours[k], "degree": deg,
                "x_km": C_hat[0], "y_km": C_hat[1], "z_km": C_hat[2],
                "sigma_range_km": np.sqrt(Pk[0, 0]), "sigma_east_km": np.sqrt(Pk[1, 1]),
                "sigma_north_km": np.sqrt(Pk[2, 2]),
                "residual_range_km": rr[0], "residual_east_km": rr[1], "residual_north_km": rr[2],
                "chi2_dof": chi2 / dof,
                "range_minus_label_km": dl[0], "east_minus_label_km": dl[1],
                "north_minus_label_km": dl[2]})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- summary

def summarize_relative(rel: pd.DataFrame, traj: pd.DataFrame | None = None,
                       opt: PoseOptions | None = None) -> dict:
    """Key numbers of the relative-pose step, for ``summary.json``.

    Parameters
    ----------
    rel : pandas.DataFrame
        Output of :func:`relative_poses`.
    traj : pandas.DataFrame, optional
        Output of :func:`fit_trajectory`.
    opt : PoseOptions, optional
        Settings used.

    Returns
    -------
    dict
    """
    ok = rel[rel.success.astype(bool)] if len(rel) else rel

    def med(col):
        return float(ok[col].median()) if col in ok.columns and ok[col].notna().any() else None

    def rms(col):
        if col not in ok.columns or not ok[col].notna().any():
            return None
        return float(np.sqrt(np.mean(np.square(ok[col].dropna()))))

    out = {"num_images": int(len(rel)), "num_images_with_pose": int(len(ok)),
           "median_inliers": med("num_inliers"), "median_rms_px": med("rms_px"),
           "median_range_km": med("range_km")}
    for k in ("range", "east", "north"):
        out[f"median_sigma_{k}_km"] = med(f"sigma_{k}_km")
    if "sigma_target_u_px" in ok.columns and len(ok):
        out["median_sigma_target_px"] = float(np.hypot(ok.sigma_target_u_px,
                                                       ok.sigma_target_v_px).median())
    for tag in ("label", "sfm"):
        for k in ("range", "east", "north"):
            out[f"rms_{k}_minus_{tag}_km"] = rms(f"{k}_minus_{tag}_km")
        out[f"mean_range_minus_{tag}_km"] = med(f"range_minus_{tag}_km")
        out[f"median_attitude_minus_{tag}_arcsec"] = med(f"attitude_minus_{tag}_arcsec")
        out[f"median_{tag}_rms_px"] = med(f"{tag}_rms_px")
    out["median_attitude_minus_ncc_arcsec"] = med("attitude_minus_ncc_arcsec")
    if traj is not None and len(traj):
        out["trajectory"] = {
            "degree": int(traj.degree.max()),
            "chi2_dof": {s: float(v) for s, v in traj.groupby("sequence").chi2_dof.first().items()},
            **{f"scatter_{k}_km": float(np.sqrt(np.mean(traj[f"residual_{k}_km"] ** 2)))
               for k in ("range", "east", "north")},
            **{f"rms_{k}_minus_label_km": float(np.sqrt(np.mean(traj[f"{k}_minus_label_km"] ** 2)))
               for k in ("range", "east", "north")},
        }
    out = {k: v for k, v in out.items() if v is not None}
    if opt is not None:
        out["options"] = asdict(opt)
    return out
