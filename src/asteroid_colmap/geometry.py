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
    """TDB - UTC in seconds: 32.184 s + TAI - UTC (leap seconds).

    The periodic TDB - TT terms (< 2 ms) are ignored.

    Parameters
    ----------
    utc : datetime.datetime
        Naive UTC epoch, 1999 or later.

    Returns
    -------
    float
        TDB - UTC (s).

    Raises
    ------
    ValueError
        If ``utc`` is before 1999, where the leap-second table starts.
    """
    if utc < _LEAP_SECONDS[0][0]:
        raise ValueError("leap-second table starts in 1999")
    tai_utc = max(n for start, n in _LEAP_SECONDS if utc >= start)
    return 32.184 + tai_utc


def tdb_days_since_j2000(utc: dt.datetime) -> float:
    """Days since J2000 (2000-01-01 12:00 TDB) on the TDB time scale.

    Parameters
    ----------
    utc : datetime.datetime
        Naive UTC epoch.

    Returns
    -------
    float
        Interval ``d`` used by the IAU rotation models (days of 86400 s).
    """
    tdb = utc + dt.timedelta(seconds=tdb_minus_utc(utc))
    return (tdb - J2000_EPOCH).total_seconds() / 86400.0


def rot_x(a: float) -> np.ndarray:
    """Frame rotation about the x axis.

    Parameters
    ----------
    a : float
        Angle (rad).

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` matrix that expresses vectors in a frame rotated by ``a`` about x
        (passive rotation: ``rot_x(a) @ v`` turns ``v`` by ``-a``).
    """
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def rot_z(a: float) -> np.ndarray:
    """Frame rotation about the z axis.

    Parameters
    ----------
    a : float
        Angle (rad).

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` matrix that expresses vectors in a frame rotated by ``a`` about z
        (passive rotation: ``rot_z(a) @ v`` turns ``v`` by ``-a``).
    """
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_matrix(q) -> np.ndarray:
    """Scalar-first quaternion -> rotation matrix (SPICE ``q2m`` / COLMAP convention).

    For a Dawn FC label QUATERNION the result rotates J2000 vectors into the camera frame;
    for a COLMAP image quaternion it rotates world vectors into the camera frame.

    Parameters
    ----------
    q : array_like
        ``(w, x, y, z)``; normalised before use.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` rotation matrix.
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
    """Rotation matrix -> scalar-first unit quaternion, the inverse of :func:`quat_to_matrix`.

    Uses Bar-Itzhack's (2000) eigenvector method, which tolerates a slightly
    non-orthogonal ``R``.

    Parameters
    ----------
    R : numpy.ndarray
        ``(3, 3)`` rotation matrix.

    Returns
    -------
    numpy.ndarray
        ``(w, x, y, z)`` with ``w >= 0``.
    """
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
    """Rotation from J2000 to the body-fixed frame at an epoch.

    ``R = Rz(W) Rx(90 deg - dec) Rz(90 deg + ra)``, with the pole ``(ra, dec)`` and the
    prime meridian angle ``W = w0 + rate * d`` of ``body`` and ``d`` from
    :func:`tdb_days_since_j2000`.

    Parameters
    ----------
    body : Body
        Rotation model.
    utc : datetime.datetime
        Naive UTC epoch.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` matrix; ``R @ v_j2000`` gives the body-fixed components of ``v``.
    """
    d = tdb_days_since_j2000(utc)
    w = np.radians((body.w0_deg + body.w_rate_deg_per_day * d) % 360.0)
    ra, dec = np.radians(body.pole_ra_deg), np.radians(body.pole_dec_deg)
    return rot_z(w) @ rot_x(np.pi / 2 - dec) @ rot_z(np.pi / 2 + ra)


def latlon_to_unit(lat_deg, lon_deg) -> np.ndarray:
    """Unit vector of a planetocentric direction.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Planetocentric latitude and east longitude (deg); broadcast together.

    Returns
    -------
    numpy.ndarray
        Unit vectors of shape ``broadcast_shape + (3,)``.
    """
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def cart_to_latlonr(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian points -> planetocentric latitude, east longitude and radius.

    Parameters
    ----------
    xyz : numpy.ndarray
        ``(N, 3)`` or ``(3,)`` body-fixed points.

    Returns
    -------
    lat_deg : numpy.ndarray
        ``(N,)`` latitude (deg).
    lon_deg : numpy.ndarray
        ``(N,)`` east longitude in [0, 360) (deg).
    radius : numpy.ndarray
        ``(N,)`` distance from the origin, in the unit of ``xyz``.
    """
    xyz = np.atleast_2d(xyz)
    r = np.linalg.norm(xyz, axis=1)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1, 1)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    return lat, lon, r


def ellipsoid_radius(lat_deg, lon_deg, radii) -> np.ndarray:
    """Radius of a triaxial ellipsoid along a planetocentric direction.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Planetocentric latitude and east longitude (deg).
    radii : sequence of float
        Semi-axes ``(a, b, c)`` along x, y and z.

    Returns
    -------
    numpy.ndarray
        Distance from the centre to the surface along each direction, unit of ``radii``.
    """
    u = latlon_to_unit(lat_deg, lon_deg)
    a, b, c = radii
    return 1.0 / np.sqrt((u[..., 0] / a) ** 2 + (u[..., 1] / b) ** 2 + (u[..., 2] / c) ** 2)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of the relative rotation ``Ra^T Rb``, i.e. the difference of two attitudes.

    Parameters
    ----------
    Ra, Rb : numpy.ndarray
        ``(3, 3)`` rotation matrices.

    Returns
    -------
    float
        Angle in [0, 180] (deg).
    """
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity ``dst ~ s R src + t`` (Umeyama 1991), proper rotation only.

    Parameters
    ----------
    src, dst : numpy.ndarray
        ``(N, 3)`` corresponding points, ``N >= 3``.
    with_scale : bool
        Estimate the scale; if False, ``s = 1`` (rigid fit).

    Returns
    -------
    s : float
        Scale.
    R : numpy.ndarray
        ``(3, 3)`` rotation with ``det(R) = +1``.
    t : numpy.ndarray
        ``(3,)`` translation, in ``dst`` units.
    """
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
    """First intersection of rays ``origin + l * dirs`` with an origin-centred ellipsoid.

    Parameters
    ----------
    origin : numpy.ndarray
        ``(3,)`` common ray origin outside the ellipsoid, in the unit of ``radii``.
    dirs : numpy.ndarray
        ``(..., 3)`` ray directions; need not be unit length.
    radii : sequence of float
        Semi-axes ``(a, b, c)`` along x, y and z.

    Returns
    -------
    points : numpy.ndarray
        ``(..., 3)`` nearest intersection; NaN where the ray misses.
    hit : numpy.ndarray
        ``(...)`` boolean mask of rays that hit the ellipsoid in front of the origin.
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
