"""Rotations, body rotation model, similarity alignment and ellipsoid ray casting."""

from __future__ import annotations

import datetime as dt

import numpy as np

from .config import Body

J2000_EPOCH = dt.datetime(2000, 1, 1, 12, 0, 0)

# TAI - UTC (leap seconds) from 1999 on; TDB - UTC = 32.184 s + (TAI - UTC) to ~2 ms.
_LEAP_SECONDS = [
    (dt.datetime(1999, 1, 1), 32),
    (dt.datetime(2006, 1, 1), 33),
    (dt.datetime(2009, 1, 1), 34),
    (dt.datetime(2012, 7, 1), 35),
    (dt.datetime(2015, 7, 1), 36),
    (dt.datetime(2017, 1, 1), 37),
]


def tdb_minus_utc(utc: dt.datetime) -> float:
    if utc < _LEAP_SECONDS[0][0]:
        raise ValueError("leap-second table starts in 1999")
    tai_utc = max(n for start, n in _LEAP_SECONDS if utc >= start)
    return 32.184 + tai_utc


def tdb_days_since_j2000(utc: dt.datetime) -> float:
    tdb = utc + dt.timedelta(seconds=tdb_minus_utc(utc))
    return (tdb - J2000_EPOCH).total_seconds() / 86400.0


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_matrix(q) -> np.ndarray:
    """Scalar-first unit quaternion -> rotation matrix (SPICE ``q2m`` / COLMAP convention).

    For a Dawn FC label QUATERNION the result rotates J2000 vectors into the camera frame;
    for a COLMAP image quaternion it rotates world vectors into the camera frame.
    """
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> scalar-first unit quaternion with w >= 0 (inverse of quat_to_matrix)."""
    R = np.asarray(R, float)
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[2, 1] - R[1, 2]],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], R[2, 1] + R[1, 2], R[0, 2] - R[2, 0]],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], R[1, 0] - R[0, 1]],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    x, y, z, w = vecs[:, np.argmax(vals)]
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


def body_from_j2000(body: Body, utc: dt.datetime) -> np.ndarray:
    """Rotation J2000 -> body-fixed: Rz(W) Rx(90 deg - dec) Rz(90 deg + ra)."""
    d = tdb_days_since_j2000(utc)
    w = np.radians((body.w0_deg + body.w_rate_deg_per_day * d) % 360.0)
    ra, dec = np.radians(body.pole_ra_deg), np.radians(body.pole_dec_deg)
    return rot_z(w) @ rot_x(np.pi / 2 - dec) @ rot_z(np.pi / 2 + ra)


def latlon_to_unit(lat_deg, lon_deg) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def cart_to_latlonr(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planetocentric latitude, east longitude in [0, 360) and radius."""
    xyz = np.atleast_2d(xyz)
    r = np.linalg.norm(xyz, axis=1)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1, 1)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    return lat, lon, r


def ellipsoid_radius(lat_deg, lon_deg, radii) -> np.ndarray:
    """Radius of a triaxial ellipsoid along a planetocentric direction."""
    u = latlon_to_unit(lat_deg, lon_deg)
    a, b, c = radii
    return 1.0 / np.sqrt((u[..., 0] / a) ** 2 + (u[..., 1] / b) ** 2 + (u[..., 2] / c) ** 2)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of the relative rotation Ra^T Rb."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity ``dst ~ s R src + t`` (Umeyama 1991), proper rotation only."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ray_ellipsoid(origin: np.ndarray, dirs: np.ndarray, radii) -> tuple[np.ndarray, np.ndarray]:
    """First intersection of rays ``origin + l * dirs`` (dirs: (..., 3)) with an ellipsoid.

    Returns (points (..., 3), hit mask (...)); misses are NaN.
    """
    r = np.asarray(radii, float)
    o = np.asarray(origin, float) / r
    d = dirs / r
    a = np.einsum("...i,...i", d, d)
    b = 2.0 * np.einsum("...i,i", d, o)
    c = o @ o - 1.0
    disc = b * b - 4 * a * c
    hit = disc > 0
    lam = np.where(hit, (-b - np.sqrt(np.where(hit, disc, 0.0))) / (2 * a), np.nan)
    hit &= lam > 0
    pts = np.asarray(origin, float) + lam[..., None] * dirs
    pts[~hit] = np.nan
    return pts, hit
