"""Find catalog landmarks in new images by normalised cross-correlation (NCC).

A catalog landmark is a 3-D point; to recognise it in an image that was not part of the
reconstruction, each curated landmark gets a small surface patch (a "maplet", as in
stereophotoclinometry):

* a local plane through the landmark, with its normal fitted to the neighbouring landmarks;
* a height grid above that plane, smoothed from the neighbouring landmarks;
* the image intensity sampled on that grid from a few catalog images, chosen so that their
  Sun directions differ as much as possible (:func:`build_templates`).

To measure a new image (:func:`match_image`):

1. predict every visible landmark with the a priori pose from the PDS label;
2. per landmark, take the maplet view whose Sun direction is closest to the new image's;
3. render it into the new image's pixel grid (each pixel's ray meets the height grid) and
   correlate it with the image over a search window (masked NCC, :func:`ncc_maps`);
4. keep strong, unambiguous peaks and fit an attitude correction (:func:`estimate_pose`);
5. render again with the corrected pose, search a smaller window and fit again.

The result is a pixel measurement of every matched landmark plus a corrected pose: the input
of landmark-based optical navigation. All poses map body-fixed vectors into the camera frame
(x right, y down, z along the boresight); pixel coordinates follow COLMAP (pixel centres at
+0.5) in the flipped image orientation used throughout the package.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .camera import FramingCamera
from .catalog import pixel_frames
from .config import Body
from .geometry import matrix_to_quat, quat_to_matrix, ray_ellipsoid, rotation_angle_deg
from .georef import pointing_offsets
from .metadata import label_geometry
from .preprocess import apply_flip, load_fits

log = logging.getLogger(__name__)

_ARCSEC = np.degrees(1.0) * 3600.0


# --------------------------------------------------------------------------- templates

@dataclass
class Templates:
    """Maplets of the curated landmarks: local surface model plus reference intensities.

    ``L`` landmarks, ``K`` view slots, ``M x M`` grid samples. Grid sample ``(row, col)`` of
    landmark ``l`` is the body-fixed point
    ``X + a e + b N + height[row, col] n`` with ``a = (col - c) s``, ``b = (row - c) s``,
    ``c = (M - 1) / 2`` and ``s = spacing_km``. View slots are filled from slot 0; unused slots
    have an empty ``view_image`` and NaN everywhere else.

    Attributes
    ----------
    landmark_id : numpy.ndarray
        ``(L,)`` catalog ids.
    X : numpy.ndarray
        ``(L, 3)`` landmark positions, body-fixed (km).
    e, N, n : numpy.ndarray
        ``(L, 3)`` unit axes of the local plane: ``e`` east-like, ``N`` north-like and the
        outward normal ``n = e x N``.
    height : numpy.ndarray
        ``(L, M, M)`` height above the plane (km); zero at the centre.
    maplet : numpy.ndarray
        ``(L, K, M, M)`` ``float16`` intensities, zero mean and unit standard deviation per view.
    view_image : numpy.ndarray
        ``(L, K)`` name of the catalog image each view was sampled from.
    view_sun : numpy.ndarray
        ``(L, K, 3)`` unit vector to the Sun at that image (body-fixed).
    view_dir : numpy.ndarray
        ``(L, K, 3)`` unit vector from the landmark to that camera.
    view_emission_deg, view_incidence_deg : numpy.ndarray
        ``(L, K)`` emission and incidence angles relative to ``n`` (deg).
    view_gsd_km : numpy.ndarray
        ``(L, K)`` pixel footprint at the landmark (km/px).
    view_contrast : numpy.ndarray
        ``(L, K)`` standard deviation over mean of the raw intensities.
    spacing_km : float
        Grid spacing ``s`` (km).
    flip : str
        Image flip the reference images were read with.
    """
    landmark_id: np.ndarray
    X: np.ndarray
    e: np.ndarray
    N: np.ndarray
    n: np.ndarray
    height: np.ndarray
    maplet: np.ndarray
    view_image: np.ndarray
    view_sun: np.ndarray
    view_dir: np.ndarray
    view_emission_deg: np.ndarray
    view_incidence_deg: np.ndarray
    view_gsd_km: np.ndarray
    view_contrast: np.ndarray
    spacing_km: float
    flip: str = "none"

    def __len__(self) -> int:
        """Number of landmarks."""
        return len(self.landmark_id)

    @property
    def size(self) -> int:
        """Maplet width ``M`` (samples)."""
        return self.height.shape[-1]

    @property
    def num_views(self) -> np.ndarray:
        """``(L,)`` number of filled view slots per landmark."""
        return (self.view_image != "").sum(axis=1)

    def save(self, path: str | Path) -> Path:
        """Write the templates as a compressed ``.npz`` file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file, usually ``<workdir>/catalog/templates.npz``.

        Returns
        -------
        pathlib.Path
            ``path``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in asdict(self).items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, spacing_km=np.float64(self.spacing_km), flip=np.str_(self.flip),
                            **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Templates:
        """Read templates written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            ``.npz`` file.

        Returns
        -------
        Templates
        """
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["spacing_km"] = float(d["spacing_km"])
        d["flip"] = str(d["flip"])
        return cls(**d)


def local_frames(X: np.ndarray, points: np.ndarray, radii, radius_km: float = 12.0,
                 min_points: int = 6, max_tilt_deg: float = 60.0):
    """Local tangent plane of the surface at each landmark.

    The normal is the smallest principal axis of the landmarks within ``radius_km``, turned
    to point away from the body. Where there are too few neighbours, or the fitted normal is
    tilted by more than ``max_tilt_deg`` from the ellipsoid normal, the ellipsoid normal is
    used instead.

    Parameters
    ----------
    X : numpy.ndarray
        ``(L, 3)`` landmark positions (km).
    points : numpy.ndarray
        ``(P, 3)`` surface points to fit (km), usually the well-graded catalog landmarks.
    radii : sequence of float
        Reference ellipsoid semi-axes (km).
    radius_km : float
        Neighbourhood radius.
    min_points : int
        Fewest neighbours for a plane fit.
    max_tilt_deg : float
        Largest accepted angle between the fitted and the ellipsoid normal.

    Returns
    -------
    e, N, n : numpy.ndarray
        ``(L, 3)`` right-handed unit axes: ``e`` is horizontal (perpendicular to the body z
        axis, east-like), ``N = n x e`` points north-like and ``n`` outward.
    """
    X = np.asarray(X, float)
    ell = X / np.asarray(radii, float) ** 2
    ell /= np.linalg.norm(ell, axis=1, keepdims=True)
    n = ell.copy()
    cos_max = np.cos(np.radians(max_tilt_deg))
    for i, idx in enumerate(cKDTree(points).query_ball_point(X, radius_km)):
        if len(idx) < min_points:
            continue
        P = points[idx] - points[idx].mean(axis=0)
        normal = np.linalg.eigh(P.T @ P)[1][:, 0]
        c = normal @ ell[i]
        if abs(c) >= cos_max:
            n[i] = normal if c > 0 else -normal
    e = np.cross([0.0, 0.0, 1.0], n)
    polar = np.linalg.norm(e, axis=1) < 1e-6
    e[polar] = np.cross([1.0, 0.0, 0.0], n[polar])
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    N = np.cross(n, e)
    return e, N, n


def height_grids(X, e, N, n, points, size: int, spacing_km: float,
                 bandwidth_km: float = 2.0) -> np.ndarray:
    """Smoothed height of the surface above each landmark's local plane.

    Heights of the neighbouring points are averaged with a Gaussian kernel
    (Nadaraya-Watson), after dropping points more than ``max(3 sigma, 1 km)`` from the
    median height (sigma from the MAD). The centre is then pinned to the landmark itself:
    the kernel estimate there is subtracted with the same Gaussian weight, so the grid
    passes through ``X`` while its surroundings are unchanged.

    Parameters
    ----------
    X, e, N, n : numpy.ndarray
        ``(L, 3)`` landmark positions and plane axes from :func:`local_frames`.
    points : numpy.ndarray
        ``(P, 3)`` surface points (km).
    size : int
        Grid width ``M`` (odd).
    spacing_km : float
        Grid spacing (km).
    bandwidth_km : float
        Kernel standard deviation (km).

    Returns
    -------
    numpy.ndarray
        ``(L, M, M)`` ``float32`` heights (km); ``[l, row, col]`` lies at
        ``a = (col - c) s`` along ``e`` and ``b = (row - c) s`` along ``N``.
    """
    c = (size - 1) / 2
    g = (np.arange(size) - c) * spacing_km
    A, B = np.meshgrid(g, g)
    pin = np.exp(-(A**2 + B**2) / (2 * bandwidth_km**2))
    radius = c * spacing_km * np.sqrt(2) + 2 * bandwidth_km
    out = np.zeros((len(X), size, size), np.float32)
    for i, idx in enumerate(cKDTree(points).query_ball_point(X, radius)):
        if not idx:
            continue
        d = points[idx] - X[i]
        a, b, h = d @ e[i], d @ N[i], d @ n[i]
        mad = 1.4826 * np.median(np.abs(h - np.median(h)))
        keep = np.abs(h - np.median(h)) <= max(3 * mad, 1.0)
        a, b, h = a[keep], b[keep], h[keep]
        w = np.exp(-((A[..., None] - a) ** 2 + (B[..., None] - b) ** 2) / (2 * bandwidth_km**2))
        H = (w @ h) / np.maximum(w.sum(axis=-1), 1e-12)
        out[i] = H - H[int(c), int(c)] * pin
    return out


def _catalog_poses(cams: pd.DataFrame, meta: pd.DataFrame, body: Body) -> dict:
    """Pose, Sun vector and FITS path of each registered catalog image.

    Uses the georeferenced COLMAP attitude and position (``sfm_q*``, ``sfm_*_km``); catalogs
    written before the attitude was exported fall back to the label attitude.

    Returns
    -------
    dict
        ``image -> (R_cam_from_bf, C_km, sun_bf, fit_path)``.
    """
    look = meta.set_index("image")
    has_q = "sfm_qw" in cams.columns
    if not has_q:
        log.warning("cameras.csv has no sfm_qw..qz (older catalog): sampling with label attitudes")
    out = {}
    for r in cams[cams.registered.astype(bool)].itertuples(index=False):
        if r.image not in look.index:
            continue
        g = label_geometry(look.loc[r.image], body)
        R = quat_to_matrix([r.sfm_qw, r.sfm_qx, r.sfm_qy, r.sfm_qz]) if has_q else g["R_cam_from_bf"]
        C = np.array([r.sfm_x_km, r.sfm_y_km, r.sfm_z_km])
        out[r.image] = (R, C, g["sun_bf"], look.loc[r.image, "fit_path"])
    return out


def _grid_points(X, e, N, n, H, spacing_km):
    """Body-fixed points ``(L, M, M, 3)`` of maplet grids with heights ``H`` ``(L, M, M)``."""
    M = H.shape[-1]
    g = (np.arange(M) - (M - 1) / 2) * spacing_km
    return (X[:, None, None, :] + g[None, None, :, None] * e[:, None, None, :]
            + g[None, :, None, None] * N[:, None, None, :] + H[..., None] * n[:, None, None, :])


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``img`` at COLMAP pixel coordinates; NaN outside the image."""
    return ndimage.map_coordinates(img, [v - 0.5, u - 0.5], order=1, mode="constant", cval=np.nan)


def build_templates(catalog_dir: str | Path, meta: pd.DataFrame, body: Body,
                    camera: FramingCamera, flip: str | None = None, size: int = 31,
                    views: int = 4, spacing_km: float | None = None,
                    max_emission_deg: float = 60.0, max_incidence_deg: float = 80.0,
                    min_sun_separation_deg: float = 3.0, normal_radius_km: float = 12.0,
                    bandwidth_km: float = 2.0, progress: bool = True) -> Templates:
    """Build a maplet for every curated landmark of a catalog.

    Reference views are chosen among the catalog images that observed the landmark, with
    emission below ``max_emission_deg``, incidence below ``max_incidence_deg`` and the whole
    maplet inside the image: first the most nadir view, then, greedily, the view whose Sun
    direction is farthest from those already chosen, until ``views`` are chosen or the next
    one would differ by less than ``min_sun_separation_deg``. Each view is sampled with the
    georeferenced COLMAP pose of its image, from the calibrated FITS file.

    Parameters
    ----------
    catalog_dir : str or pathlib.Path
        Catalog folder with ``landmarks.csv``, ``landmarks_curated.csv``, ``observations.csv``,
        ``cameras.csv`` and ``summary.json``.
    meta : pandas.DataFrame
        Metadata of the catalog images (``fit_path`` and the label geometry).
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    flip : str, optional
        Image flip; defaults to ``image_flip`` in ``summary.json``.
    size : int
        Maplet width ``M`` in samples (odd).
    views : int
        Maximum number of reference views per landmark.
    spacing_km : float, optional
        Grid spacing; defaults to the median ground sample distance of the curated landmarks.
    max_emission_deg, max_incidence_deg : float
        Reference-view limits, relative to the local normal.
    min_sun_separation_deg : float
        Smallest Sun-direction difference that justifies another view.
    normal_radius_km : float
        Neighbourhood radius of the plane fit (:func:`local_frames`).
    bandwidth_km : float
        Height smoothing (:func:`height_grids`).
    progress : bool
        Show a progress bar.

    Returns
    -------
    Templates
        Landmarks without any usable view are left out.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd")
    cat = Path(catalog_dir)
    lm = pd.read_csv(cat / "landmarks.csv")
    cur = pd.read_csv(cat / "landmarks_curated.csv")
    cams = pd.read_csv(cat / "cameras.csv")
    obs = pd.read_csv(cat / "observations.csv", usecols=["landmark_id", "image"])
    if flip is None:
        summary = cat / "summary.json"
        flip = json.loads(summary.read_text()).get("image_flip", "none") if summary.exists() else "none"
    s = float(spacing_km or round(float(np.median(cur.gsd_km)), 3))
    M, c = size, (size - 1) / 2

    good = lm[~lm.outlier.astype(bool) & lm.grade.isin(["A", "B"])]
    support = good[["x_km", "y_km", "z_km"]].to_numpy()
    X = cur[["x_km", "y_km", "z_km"]].to_numpy()
    e, N, n = local_frames(X, support, body.radii_km, radius_km=normal_radius_km)
    H = height_grids(X, e, N, n, support, M, s, bandwidth_km=bandwidth_km)
    log.info("maplet frames and heights for %d landmarks (%d x %d at %.3f km)", len(X), M, M, s)

    # candidate views: (landmark, image) observations passing the angle and footprint tests
    poses = _catalog_poses(cams, meta, body)
    li_of = pd.Series(np.arange(len(cur)), index=cur.landmark_id)
    obs = obs[obs.landmark_id.isin(li_of.index) & obs.image.isin(poses)]
    li = li_of.loc[obs.landmark_id].to_numpy()
    img_names = obs.image.to_numpy()
    C = np.stack([poses[i][1] for i in img_names])
    sun = np.stack([poses[i][2] for i in img_names])
    d = C - X[li]
    rng = np.linalg.norm(d, axis=1)
    vdir = d / rng[:, None]
    emi = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", vdir, n[li]), -1, 1)))
    inc = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", sun, n[li]), -1, 1)))
    ok = (emi < max_emission_deg) & (inc < max_incidence_deg)
    corners = _grid_points(X[li], e[li], N[li], n[li], H[:, ::M - 1, ::M - 1][li], s * (M - 1))
    for name in np.unique(img_names):
        sel = np.flatnonzero(img_names == name)
        R, Cc = poses[name][:2]
        uv = camera.project((corners[sel].reshape(-1, 3) - Cc) @ R.T).reshape(len(sel), 4, 2)
        inside = ((uv >= 2) & (uv <= np.array([camera.width, camera.height]) - 2)).all(axis=(1, 2))
        ok[sel] &= inside
    cand = pd.DataFrame({"li": li, "image": img_names, "emission": emi, "incidence": inc,
                         "gsd": rng * camera.ifov_rad})[ok]
    cand_sun, cand_dir = sun[ok], vdir[ok]

    # greedy, Sun-diverse choice per landmark
    chosen = []  # (li, slot, row in cand)
    cos_sep = np.cos(np.radians(min_sun_separation_deg))
    for l, rows in cand.groupby("li").indices.items():
        first = rows[np.argmin(cand.emission.to_numpy()[rows])]
        picked = [first]
        while len(picked) < views:
            sim = (cand_sun[rows] @ cand_sun[picked].T).max(axis=1)  # cos of nearest chosen Sun
            j = int(np.argmin(sim))
            if sim[j] > cos_sep:
                break
            picked.append(rows[j])
        chosen += [(l, k, r) for k, r in enumerate(picked)]
    chosen = np.array(chosen, int).reshape(-1, 3)
    log.info("%d reference views for %d landmarks", len(chosen), len(np.unique(chosen[:, 0])))

    L, K = len(cur), views
    maplet = np.full((L, K, M, M), np.nan, np.float16)
    view_image = np.full((L, K), "", dtype=object)
    vsun, vdirs = np.full((L, K, 3), np.nan), np.full((L, K, 3), np.nan)
    vemi, vinc, vgsd, vcon = (np.full((L, K), np.nan) for _ in range(4))
    cimg = cand.image.to_numpy()
    groups = pd.Series(np.arange(len(chosen))).groupby(cimg[chosen[:, 2]]).indices
    for name, idx in tqdm(groups.items(), desc="templates", unit="image", disable=not progress):
        R, Cc, _, fit_path = poses[name]
        img = apply_flip(load_fits(fit_path), flip)
        dark = 0.02 * np.percentile(img, 99.5)
        l_, k_, r_ = chosen[idx].T
        P = _grid_points(X[l_], e[l_], N[l_], n[l_], H[l_], s)
        uv = camera.project((P.reshape(-1, 3) - Cc) @ R.T)
        vals = _sample(img, uv[:, 0], uv[:, 1]).reshape(len(idx), M, M)
        mean = vals.mean(axis=(1, 2))
        std = vals.std(axis=(1, 2))
        usable = (np.isfinite(mean) & (mean > 0) & (std > 0.01 * np.abs(mean))
                  & ((vals < dark).mean(axis=(1, 2)) < 0.3))
        for j in np.flatnonzero(usable):
            lj, kj, rj = l_[j], k_[j], r_[j]
            maplet[lj, kj] = (vals[j] - mean[j]) / std[j]
            view_image[lj, kj] = name
            vsun[lj, kj], vdirs[lj, kj] = cand_sun[rj], cand_dir[rj]
            vemi[lj, kj] = cand.emission.iat[rj]
            vinc[lj, kj] = cand.incidence.iat[rj]
            vgsd[lj, kj] = cand.gsd.iat[rj]
            vcon[lj, kj] = std[j] / mean[j]

    # move filled slots to the front and drop landmarks without a view
    order = np.argsort(view_image == "", axis=1, kind="stable")
    take = lambda a: np.take_along_axis(a, order.reshape(order.shape + (1,) * (a.ndim - 2)), axis=1)
    maplet, view_image, vsun, vdirs = take(maplet), take(view_image), take(vsun), take(vdirs)
    vemi, vinc, vgsd, vcon = take(vemi), take(vinc), take(vgsd), take(vcon)
    keep = view_image[:, 0] != ""
    log.info("%d of %d curated landmarks have templates (%.2f views each)",
             keep.sum(), L, (view_image[keep] != "").sum(axis=1).mean() if keep.any() else 0)
    return Templates(
        landmark_id=cur.landmark_id.to_numpy().astype(str)[keep], X=X[keep], e=e[keep], N=N[keep],
        n=n[keep], height=H[keep], maplet=maplet[keep], view_image=view_image[keep].astype(str),
        view_sun=vsun[keep], view_dir=vdirs[keep], view_emission_deg=vemi[keep],
        view_incidence_deg=vinc[keep], view_gsd_km=vgsd[keep], view_contrast=vcon[keep],
        spacing_km=s, flip=flip)


# --------------------------------------------------------------------------- rendering and NCC

def render_templates(t: Templates, idx: np.ndarray, view: np.ndarray, R: np.ndarray,
                     C: np.ndarray, camera: FramingCamera, col0: np.ndarray, row0: np.ndarray,
                     size: int, maplet_sigma: float = 0.0, iterations: int = 3):
    """Render maplets into windows of a new image.

    Each window pixel's ray is intersected with the landmark's height grid (a plane at the
    current height estimate, ``iterations`` times) and the maplet is sampled bilinearly
    there.

    Parameters
    ----------
    t : Templates
        Maplets.
    idx, view : numpy.ndarray
        ``(m,)`` landmark indices and the view slot to render for each.
    R, C : numpy.ndarray
        Pose of the new image: body-fixed -> camera rotation and camera position (km).
    camera : FramingCamera
        Camera model.
    col0, row0 : numpy.ndarray
        ``(m,)`` integer image column and row of each window's top-left pixel (0-based
        array indices of the flipped image).
    size : int
        Window width ``T`` (px).
    maplet_sigma : float
        Gaussian blur of the maplets before sampling (samples), for images coarser than the
        maplet spacing.
    iterations : int
        Ray/height-grid iterations.

    Returns
    -------
    values : numpy.ndarray
        ``(m, T, T)`` rendered intensities; NaN outside the maplet.
    valid : numpy.ndarray
        ``(m, T, T)`` mask of pixels inside the maplet.
    """
    m, T = len(idx), size
    j = np.arange(T) + 0.5
    u = np.broadcast_to(col0[:, None, None] + j[None, None, :], (m, T, T))
    v = np.broadcast_to(row0[:, None, None] + j[None, :, None], (m, T, T))
    d = camera.pixel_rays(u, v) @ R  # body-fixed ray directions
    CX = C - t.X[idx]
    e, N, n = t.e[idx], t.N[idx], t.n[idx]
    nd = np.einsum("mijk,mk->mij", d, n)
    ed = np.einsum("mijk,mk->mij", d, e)
    Nd = np.einsum("mijk,mk->mij", d, N)
    nc = np.einsum("mk,mk->m", CX, n)[:, None, None]
    ec = np.einsum("mk,mk->m", CX, e)[:, None, None]
    Nc = np.einsum("mk,mk->m", CX, N)[:, None, None]
    s, c, M = t.spacing_km, (t.size - 1) / 2, t.size
    li = np.broadcast_to(np.arange(m)[:, None, None], (m, T, T))
    H = t.height[idx].astype(np.float64)
    h = np.zeros((m, T, T))
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iterations):
            lam = (h - nc) / nd
            col = (ec + lam * ed) / s + c
            row = (Nc + lam * Nd) / s + c
            h = ndimage.map_coordinates(H, [li, row, col], order=1, mode="nearest")
    maps = t.maplet[idx, view].astype(np.float32)
    if maplet_sigma > 0:
        maps = ndimage.gaussian_filter(maps, (0, maplet_sigma, maplet_sigma), mode="nearest")
    vals = ndimage.map_coordinates(maps, [li, row, col], order=1, mode="constant", cval=np.nan)
    valid = (row >= 0) & (row <= M - 1) & (col >= 0) & (col <= M - 1) & np.isfinite(vals)
    return np.where(valid, vals, np.nan), valid


def ncc_maps(regions: np.ndarray, templates: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
    """Masked normalised cross-correlation of templates over search regions.

    For every shift, the correlation uses only the template pixels in ``masks``; means and
    variances of the image window are taken over the same pixels. All sums are computed
    with FFTs, batched over the first axis.

    Parameters
    ----------
    regions : numpy.ndarray
        ``(m, W, W)`` image regions.
    templates : numpy.ndarray
        ``(m, T, T)`` templates (``T <= W``); NaN pixels are treated as masked.
    masks : numpy.ndarray, optional
        ``(m, T, T)`` boolean masks; default: the finite template pixels.

    Returns
    -------
    numpy.ndarray
        ``(m, W - T + 1, W - T + 1)`` NCC in ``[-1, 1]``; element ``[k, dy, dx]`` compares
        the template with ``regions[k, dy:dy + T, dx:dx + T]``. NaN where the template or the
        window has no variance.
    """
    m, W = regions.shape[0], regions.shape[-1]
    T = templates.shape[-1]
    valid = np.isfinite(templates) if masks is None else masks & np.isfinite(templates)
    w = valid.astype(np.float64)
    cnt = w.sum(axis=(1, 2))[:, None, None]
    t = np.where(valid, templates, 0.0)
    t = (t - t.sum(axis=(1, 2))[:, None, None] / np.maximum(cnt, 1)) * w
    tnorm = np.sqrt((t**2).sum(axis=(1, 2)))[:, None, None]
    f = np.asarray(regions, np.float64)
    mu = f.mean(axis=(1, 2), keepdims=True)
    sd = f.std(axis=(1, 2), keepdims=True)
    f = (f - mu) / np.where(sd > 0, sd, 1.0)
    Ff, Ff2 = np.fft.rfft2(f), np.fft.rfft2(f * f)
    n = W - T + 1

    def corr(F, k):
        return np.fft.irfft2(F * np.conj(np.fft.rfft2(k, s=(W, W))), s=(W, W))[:, :n, :n]

    num = corr(Ff, t)
    s1, s2 = corr(Ff, w), corr(Ff2, w)
    var = s2 - s1**2 / np.maximum(cnt, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / (tnorm * np.sqrt(var))
    out[~((var > 1e-9 * cnt) & (tnorm > 0))] = np.nan
    return np.clip(out, -1.0, 1.0)


def ncc_peaks(maps: np.ndarray, exclusion_px: float = 3.0):
    """Best peak, sub-pixel position and runner-up of each NCC map.

    Parameters
    ----------
    maps : numpy.ndarray
        ``(m, S, S)`` NCC maps from :func:`ncc_maps`.
    exclusion_px : float
        The runner-up is the highest local maximum farther than this from the best peak.

    Returns
    -------
    dict of numpy.ndarray
        ``dx``, ``dy``: peak position in map coordinates, refined by a parabola through the
        peak and its two neighbours on each axis (NaN when the peak is on the border);
        ``ncc``: peak value; ``second``: runner-up value (-1 if none); ``interior``: peak not
        on the border.
    """
    m, S = maps.shape[0], maps.shape[-1]
    z = np.where(np.isfinite(maps), maps, -1.0)
    k = np.argmax(z.reshape(m, -1), axis=1)
    py, px = np.divmod(k, S)
    r = np.arange(m)
    best = z[r, py, px]
    interior = (py > 0) & (py < S - 1) & (px > 0) & (px < S - 1) & np.isfinite(maps[r, py, px])
    pyc, pxc = np.clip(py, 1, S - 2), np.clip(px, 1, S - 2)

    def vertex(lo, mid, hi):
        den = lo - 2 * mid + hi
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(den < 0, 0.5 * (lo - hi) / den, 0.0)
        return np.clip(off, -0.5, 0.5)

    dx = pxc + vertex(z[r, pyc, pxc - 1], z[r, pyc, pxc], z[r, pyc, pxc + 1])
    dy = pyc + vertex(z[r, pyc - 1, pxc], z[r, pyc, pxc], z[r, pyc + 1, pxc])
    dx[~interior], dy[~interior] = np.nan, np.nan
    peaks = z == ndimage.maximum_filter(z, size=(1, 3, 3), mode="nearest")
    yy, xx = np.mgrid[:S, :S]
    far = (yy[None] - py[:, None, None]) ** 2 + (xx[None] - px[:, None, None]) ** 2 > exclusion_px**2
    second = np.where(peaks & far, z, -1.0).reshape(m, -1).max(axis=1)
    return {"dx": dx, "dy": dy, "ncc": best, "second": second, "interior": interior}


# --------------------------------------------------------------------------- pose fit

@dataclass
class PoseFit:
    """Result of :func:`estimate_pose`.

    Attributes
    ----------
    R : numpy.ndarray
        ``(3, 3)`` estimated body-fixed -> camera rotation.
    C : numpy.ndarray
        ``(3,)`` estimated camera position (km); the a priori one unless the position is fitted.
    rotvec : numpy.ndarray
        ``(3,)`` attitude correction as a rotation vector in the camera frame (rad):
        ``R = Rotation.from_rotvec(rotvec) @ R0``.
    inliers : numpy.ndarray
        Boolean mask of the measurements kept.
    prefit_rms_px, postfit_rms_px : float
        RMS pixel residual of the inliers with the a priori and the estimated pose.
    sigma_rot_arcsec : numpy.ndarray
        ``(3,)`` 1-sigma of ``rotvec`` about the camera x, y and z axes (arcsec).
    sigma_pos_km : numpy.ndarray
        ``(3,)`` 1-sigma of the position (km); NaN when the position is not fitted.
    success : bool
        False when too few measurements were available; ``R``, ``C`` are then the a priori.
    covariance : numpy.ndarray
        ``(3, 3)``, or ``(6, 6)`` with the position, covariance of ``rotvec`` (rad) and the
        position correction (km, body-fixed).
    """
    R: np.ndarray
    C: np.ndarray
    rotvec: np.ndarray
    inliers: np.ndarray
    prefit_rms_px: float
    postfit_rms_px: float
    sigma_rot_arcsec: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    sigma_pos_km: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    success: bool = True
    covariance: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))


def _rms(r: np.ndarray) -> float:
    """Root mean square of the row norms of ``r`` (NaN when empty)."""
    return float(np.sqrt(np.mean(np.sum(r * r, axis=-1)))) if len(r) else float("nan")


def estimate_pose(X: np.ndarray, uv: np.ndarray, R0: np.ndarray, C0: np.ndarray,
                  camera: FramingCamera, sigma_px: float = 0.3, estimate_position: bool = False,
                  position_sigma_km: float | None = 10.0, k_sigma: float = 3.0,
                  floor_px: float = 1.0, min_points: int = 6, rounds: int = 3) -> PoseFit:
    """Fit an attitude (and optionally position) correction to landmark measurements.

    Minimises the pixel residuals, scaled by ``sigma_px``, with a soft-L1 loss; the position
    correction, if fitted, has a Gaussian prior of ``position_sigma_km`` (none when it is
    ``None``: a full 6-DOF solution, see :mod:`~asteroid_colmap.pose`). After each round,
    measurements whose residual exceeds ``max(median + k_sigma * 1.4826 MAD, floor_px)`` are
    dropped and the fit is repeated. The covariance is ``(J^T J)^-1`` scaled by the
    a posteriori variance factor when it exceeds one.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, 3)`` landmark positions, body-fixed (km).
    uv : numpy.ndarray
        ``(n, 2)`` measured pixel positions.
    R0, C0 : numpy.ndarray
        A priori attitude (body-fixed -> camera) and position (km).
    camera : FramingCamera
        Camera model.
    sigma_px : float
        Measurement noise (px).
    estimate_position : bool
        Also fit a position correction.
    position_sigma_km : float or None
        Prior 1-sigma of the position correction; ``None`` for no prior.
    k_sigma, floor_px : float
        Outlier threshold (see above).
    min_points : int
        Fewest inliers for a fit.
    rounds : int
        Fit/reject rounds.

    Returns
    -------
    PoseFit
    """
    X, uv = np.asarray(X, float), np.asarray(uv, float)
    npar = 6 if estimate_position else 3

    def pose(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix() @ R0
        return R, (C0 + p[3:6] if estimate_position else C0)

    def project(p):
        R, C = pose(p)
        return camera.project((X - C) @ R.T)

    prior = estimate_position and position_sigma_km is not None

    def fun(p, keep):
        r = ((project(p) - uv)[keep] / sigma_px).ravel()
        return np.concatenate([r, p[3:6] / position_sigma_km]) if prior else r

    keep = np.isfinite(uv).all(axis=1)
    p = np.zeros(npar)
    if keep.sum() < min_points:
        return PoseFit(R0, C0, np.zeros(3), keep & False, float("nan"), float("nan"), success=False)
    sol = None
    for _ in range(rounds):
        sol = least_squares(fun, p, args=(keep,), loss="soft_l1", f_scale=3.0, x_scale="jac")
        p = sol.x
        res = np.linalg.norm(project(p) - uv, axis=1)
        med = np.median(res[keep])
        thr = max(med + k_sigma * 1.4826 * np.median(np.abs(res[keep] - med)), floor_px)
        new = np.isfinite(res) & (res <= thr)
        if new.sum() < min_points:
            break
        if (new == keep).all():
            break
        keep = new
        sol = least_squares(fun, p, args=(keep,), loss="soft_l1", f_scale=3.0, x_scale="jac")
        p = sol.x
    R, C = pose(p)
    r_post = (project(p) - uv)[keep]
    r_pre = (project(np.zeros(npar)) - uv)[keep]
    J = sol.jac
    dof = max(2 * keep.sum() - npar, 1)
    vf = max(1.0, float(np.sum((r_post / sigma_px) ** 2)) / dof)
    try:
        cov = np.linalg.inv(J.T @ J) * vf
    except np.linalg.LinAlgError:
        cov = np.full((npar, npar), np.nan)
    sig = np.sqrt(np.diag(cov))
    return PoseFit(R, C, p[:3].copy(), keep, _rms(r_pre), _rms(r_post),
                   sigma_rot_arcsec=sig[:3] * _ARCSEC,
                   sigma_pos_km=sig[3:6] if estimate_position else np.full(3, np.nan),
                   success=bool(keep.sum() >= min_points), covariance=cov)


# --------------------------------------------------------------------------- matching

@dataclass
class MatchOptions:
    """Settings of :func:`match_image`.

    Attributes
    ----------
    max_emission_deg, max_incidence_deg : float
        Visibility limits relative to the maplet normal.
    max_sun_angle_deg : float
        Largest angle between the new image's Sun direction and the chosen view's.
    template_px : int
        Largest rendered template (px, odd); smaller when the maplet does not cover it.
    min_template_px : int
        Smallest useful template (px).
    min_valid_fraction : float
        Fewest template pixels that must fall on the maplet.
    coarse_search_px, coarse_landmarks : int
        Pass 1: half-width of the search window and number of high-contrast landmarks;
        ``coarse_search_px <= search_px`` skips the pass.
    search_px : int
        Pass 2: half-width of the search window around the prediction.
    refine_search_px : int
        Pass 3: half-width after the first pose fit.
    min_ncc : float
        Peak correlation needed to accept a match.
    min_margin : float
        Lead of the peak over the runner-up needed to accept a match.
    sigma_px : float
        Measurement noise for the pose fit (px).
    estimate_position : bool
        Fit a position correction too (weakly observable from far away).
    position_sigma_km : float
        Prior 1-sigma of the position correction.
    min_matches : int
        Fewest accepted matches for a pose fit.
    occlusion_margin_km : float
        A landmark counts as hidden when the ray from the camera meets the reference
        ellipsoid this much before reaching it.
    batch : int
        Landmarks rendered at once (memory).
    """
    max_emission_deg: float = 70.0
    max_incidence_deg: float = 85.0
    max_sun_angle_deg: float = 30.0
    template_px: int = 25
    min_template_px: int = 11
    min_valid_fraction: float = 0.6
    coarse_search_px: int = 20
    coarse_landmarks: int = 160
    search_px: int = 6
    refine_search_px: int = 3
    min_ncc: float = 0.8
    min_margin: float = 0.1
    sigma_px: float = 0.3
    estimate_position: bool = False
    position_sigma_km: float = 10.0
    min_matches: int = 10
    occlusion_margin_km: float = 25.0
    batch: int = 400


def visible_landmarks(t: Templates, R: np.ndarray, C: np.ndarray, sun: np.ndarray,
                      camera: FramingCamera, margin_px: float, max_emission_deg: float = 70.0,
                      max_incidence_deg: float = 85.0, radii=None,
                      occlusion_margin_km: float = 25.0) -> dict:
    """Landmarks that are lit, face the camera and project inside the image.

    Parameters
    ----------
    t : Templates
        Maplets.
    R, C : numpy.ndarray
        Pose (body-fixed -> camera rotation, camera position in km).
    sun : numpy.ndarray
        ``(3,)`` unit vector to the Sun, body-fixed.
    camera : FramingCamera
        Camera model.
    margin_px : float
        Distance the prediction must keep from the image border.
    max_emission_deg, max_incidence_deg : float
        Angle limits relative to the maplet normal.
    radii : sequence of float, optional
        Reference ellipsoid; if given, landmarks behind the body's limb are dropped.
    occlusion_margin_km : float
        See :class:`MatchOptions`.

    Returns
    -------
    dict of numpy.ndarray
        ``idx`` (indices into ``t``), ``uv`` (predicted pixels), ``range_km``,
        ``emission_deg`` and ``incidence_deg``, for the visible landmarks only.
    """
    d = C - t.X
    rng = np.linalg.norm(d, axis=1)
    emi = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d / rng[:, None], t.n), -1, 1)))
    inc = np.degrees(np.arccos(np.clip(t.n @ sun, -1, 1)))
    Xc = (t.X - C) @ R.T
    uv = camera.project(Xc)
    lo = margin_px
    ok = ((Xc[:, 2] > 0) & (emi < max_emission_deg) & (inc < max_incidence_deg)
          & (uv[:, 0] >= lo) & (uv[:, 0] <= camera.width - lo)
          & (uv[:, 1] >= lo) & (uv[:, 1] <= camera.height - lo))
    if radii is not None and ok.any():
        pts, hit = ray_ellipsoid(C, -d[ok], radii)
        first = np.where(hit, np.linalg.norm(pts - C, axis=1), np.inf)
        ok[np.flatnonzero(ok)[first < rng[ok] - occlusion_margin_km]] = False
    i = np.flatnonzero(ok)
    return {"idx": i, "uv": uv[i], "range_km": rng[i], "emission_deg": emi[i], "incidence_deg": inc[i]}


def choose_views(t: Templates, idx: np.ndarray, sun: np.ndarray, exclude_image: str | None = None):
    """Per landmark, the view slot whose Sun direction is closest to ``sun``.

    Parameters
    ----------
    t : Templates
        Maplets.
    idx : numpy.ndarray
        ``(m,)`` landmark indices.
    sun : numpy.ndarray
        ``(3,)`` unit vector to the Sun, body-fixed.
    exclude_image : str, optional
        Ignore views sampled from this image (the image being matched, if it is a catalog
        image).

    Returns
    -------
    view : numpy.ndarray
        ``(m,)`` slot index; -1 when no view is left.
    angle_deg : numpy.ndarray
        ``(m,)`` angle between the two Sun directions (deg; NaN without a view).
    """
    cos = np.einsum("lkj,j->lk", t.view_sun[idx], sun)
    usable = t.view_image[idx] != ""
    if exclude_image is not None:
        usable &= t.view_image[idx] != exclude_image
    cos = np.where(usable, cos, -2.0)
    view = np.argmax(cos, axis=1)
    best = cos[np.arange(len(idx)), view]
    angle = np.degrees(np.arccos(np.clip(best, -1, 1)))
    view[best < -1.5] = -1
    angle[best < -1.5] = np.nan
    return view, angle


def template_size(t: Templates, gsd_km: float, largest: int = 25) -> int:
    """Largest odd template width whose corners stay on the maplet when seen face-on.

    Parameters
    ----------
    t : Templates
        Maplets (spacing and size).
    gsd_km : float
        Pixel footprint of the new image at the landmarks (km/px).
    largest : int
        Upper limit (odd).

    Returns
    -------
    int
        ``min(largest, 2 floor(half / (gsd sqrt 2)) + 1)`` with ``half`` the maplet half-width.
    """
    half_km = (t.size - 1) / 2 * t.spacing_km
    return int(min(largest, 2 * np.floor(half_km / (gsd_km * np.sqrt(2))) + 1))


def _measure(img, t, idx, view, R, C, camera, T, search, maplet_sigma, opt):
    """Render, correlate and locate a set of landmarks around their predictions.

    Returns a dict of ``(m,)`` arrays: predicted ``u_pred``/``v_pred``, measured ``u``/``v``
    (NaN when the peak is on the window border or the window leaves the image), ``ncc``,
    ``ncc_second`` and ``valid_fraction``.
    """
    m = len(idx)
    out = {k: np.full(m, np.nan) for k in ("u_pred", "v_pred", "u", "v", "ncc", "ncc_second",
                                           "valid_fraction")}
    if m == 0:
        return out
    uv = camera.project((t.X[idx] - C) @ R.T)
    out["u_pred"], out["v_pred"] = uv[:, 0], uv[:, 1]
    h, W = T // 2, T + 2 * search
    col0 = np.round(uv[:, 0] - 0.5).astype(int) - h
    row0 = np.round(uv[:, 1] - 0.5).astype(int) - h
    inside = ((col0 - search >= 0) & (row0 - search >= 0)
              & (col0 - search + W <= img.shape[1]) & (row0 - search + W <= img.shape[0]))
    ar = np.arange(W)
    for s0 in range(0, m, opt.batch):
        b = np.arange(s0, min(s0 + opt.batch, m))
        b = b[inside[b]]
        if not len(b):
            continue
        tmpl, valid = render_templates(t, idx[b], view[b], R, C, camera, col0[b], row0[b], T,
                                       maplet_sigma=maplet_sigma)
        frac = valid.mean(axis=(1, 2))
        rows = row0[b, None] - search + ar
        cols = col0[b, None] - search + ar
        reg = img[rows[:, :, None], cols[:, None, :]]
        pk = ncc_peaks(ncc_maps(reg, tmpl, valid))
        good = frac >= opt.min_valid_fraction
        out["valid_fraction"][b] = frac
        out["ncc"][b] = np.where(good, pk["ncc"], np.nan)
        out["ncc_second"][b] = np.where(good, pk["second"], np.nan)
        out["u"][b] = np.where(good, uv[b, 0] + pk["dx"] - search, np.nan)
        out["v"][b] = np.where(good, uv[b, 1] + pk["dy"] - search, np.nan)
    return out


def _accepted(meas: dict, opt: MatchOptions) -> np.ndarray:
    """Matches with a finite position, a high peak and a clear lead over the runner-up."""
    with np.errstate(invalid="ignore"):
        return (np.isfinite(meas["u"]) & (meas["ncc"] >= opt.min_ncc)
                & (meas["ncc"] - meas["ncc_second"] >= opt.min_margin))


def _spread(uv: np.ndarray, score: np.ndarray, n: int, width: int, height: int, cells: int = 4):
    """Indices of up to ``n`` high-score points spread over a ``cells x cells`` image grid."""
    cell = (np.clip(uv[:, 1] / height * cells, 0, cells - 1).astype(int) * cells
            + np.clip(uv[:, 0] / width * cells, 0, cells - 1).astype(int))
    per = int(np.ceil(n / cells**2))
    order = np.lexsort((-score, cell))
    rank = np.arange(len(order)) - np.searchsorted(cell[order], cell[order])
    pick = order[rank < per]
    return pick[np.argsort(-score[pick])][:n]


def match_image(img: np.ndarray, row, t: Templates, body: Body, camera: FramingCamera,
                opt: MatchOptions | None = None, R0: np.ndarray | None = None,
                C0: np.ndarray | None = None, exclude_image: str | None = None):
    """Measure the catalog landmarks in one image and correct its pose.

    Parameters
    ----------
    img : numpy.ndarray
        Image in the package's orientation (FITS array after :func:`apply_flip`).
    row : pandas.Series or dict
        Metadata row of the image (label geometry; see
        :func:`~asteroid_colmap.metadata.label_geometry`).
    t : Templates
        Maplets.
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    opt : MatchOptions, optional
        Settings.
    R0, C0 : numpy.ndarray, optional
        A priori pose; defaults to the label attitude and position.
    exclude_image : str, optional
        Do not use reference views from this catalog image.

    Returns
    -------
    matches : pandas.DataFrame
        One row per landmark that was searched for: ``landmark_id``, ``view_image``,
        ``sun_angle_deg``, ``u_apriori``/``v_apriori`` (prediction with the a priori pose),
        ``u``/``v`` (NCC measurement, final pass), ``ncc``, ``ncc_second``, ``accepted``,
        ``inlier``, ``residual_u_px``/``residual_v_px`` (projection with the estimated pose
        minus measurement), ``emission_deg``, ``incidence_deg``, ``range_km``,
        ``template_px`` and ``valid_fraction``.
    info : dict
        Pose and statistics: ``R``, ``C`` (estimate), ``R0``, ``C0``, ``fit``
        (:class:`PoseFit`), ``num_visible``, ``num_searched``, ``num_accepted``,
        ``coarse_matches``, ``coarse_shift_px``, ``gsd_km``, ``template_px``.
    """
    opt = opt or MatchOptions()
    g = label_geometry(row, body)
    R0 = g["R_cam_from_bf"] if R0 is None else R0
    C0 = g["sc_bf"] if C0 is None else C0
    sun = g["sun_bf"]

    # footprint and template size from the landmarks near the boresight
    rng0 = np.linalg.norm(t.X - C0, axis=1)
    gsd = float(np.percentile(rng0, 5)) * camera.ifov_rad
    T = template_size(t, gsd, opt.template_px)
    info = {"R0": R0, "C0": C0, "gsd_km": gsd, "template_px": T, "num_visible": 0,
            "num_searched": 0, "num_accepted": 0, "coarse_matches": 0,
            "coarse_shift_px": (np.nan, np.nan)}
    if T < opt.min_template_px:
        raise ValueError(f"image footprint {gsd:.3f} km/px is too fine for {t.spacing_km:.3f} km maplets")
    ratio = t.spacing_km / gsd
    work = img.astype(np.float64)
    maplet_sigma = 0.0
    if ratio > 1:  # image finer than the maplets: blur the image
        work = ndimage.gaussian_filter(work, 0.5 * np.sqrt(ratio**2 - 1))
    elif ratio < 1:  # image coarser: blur the maplets
        maplet_sigma = 0.5 * np.sqrt(1 / ratio**2 - 1)

    vis = visible_landmarks(t, R0, C0, sun, camera, T // 2 + opt.search_px + 2,
                            opt.max_emission_deg, opt.max_incidence_deg, body.radii_km,
                            opt.occlusion_margin_km)
    view, sun_ang = choose_views(t, vis["idx"], sun, exclude_image)
    use = (view >= 0) & (sun_ang <= opt.max_sun_angle_deg)
    idx, view = vis["idx"][use], view[use]
    info["num_visible"] = len(vis["idx"])
    info["num_searched"] = len(idx)
    R, C = R0, C0

    # pass 1: a wide search on a few high-contrast landmarks, attitude-only fit
    if opt.coarse_search_px > opt.search_px and len(idx):
        lo = T // 2 + opt.coarse_search_px + 2
        uv = vis["uv"][use]
        inner = np.flatnonzero((uv >= lo).all(axis=1) & (uv[:, 0] <= camera.width - lo)
                               & (uv[:, 1] <= camera.height - lo))
        score = t.view_contrast[idx[inner], view[inner]]
        pick = inner[_spread(uv[inner], score, opt.coarse_landmarks, camera.width, camera.height)]
        m1 = _measure(work, t, idx[pick], view[pick], R, C, camera, T, opt.coarse_search_px,
                      maplet_sigma, opt)
        ok = _accepted(m1, opt)
        info["coarse_matches"] = int(ok.sum())
        if ok.sum() >= opt.min_matches:
            shift = np.column_stack([m1["u"] - m1["u_pred"], m1["v"] - m1["v_pred"]])[ok]
            info["coarse_shift_px"] = tuple(np.median(shift, axis=0))
            fit1 = estimate_pose(t.X[idx[pick]][ok], np.column_stack([m1["u"], m1["v"]])[ok],
                                 R, C, camera, sigma_px=opt.sigma_px, floor_px=2.0)
            if fit1.success:
                R = fit1.R

    # pass 2: every landmark, then a pose fit
    m2 = _measure(work, t, idx, view, R, C, camera, T, opt.search_px, maplet_sigma, opt)
    ok2 = _accepted(m2, opt)
    fit = estimate_pose(t.X[idx][ok2], np.column_stack([m2["u"], m2["v"]])[ok2], R0, C0, camera,
                        sigma_px=opt.sigma_px, estimate_position=opt.estimate_position,
                        position_sigma_km=opt.position_sigma_km, min_points=opt.min_matches)
    meas, acc = m2, ok2
    if fit.success:
        # pass 3: render with the fitted pose, narrow search, fit again
        m3 = _measure(work, t, idx, view, fit.R, fit.C, camera, T, opt.refine_search_px,
                      maplet_sigma, opt)
        with np.errstate(invalid="ignore"):
            ok3 = ok2 & np.isfinite(m3["u"]) & (m3["ncc"] >= opt.min_ncc)
        fit3 = estimate_pose(t.X[idx][ok3], np.column_stack([m3["u"], m3["v"]])[ok3], R0, C0,
                             camera, sigma_px=opt.sigma_px, estimate_position=opt.estimate_position,
                             position_sigma_km=opt.position_sigma_km, min_points=opt.min_matches)
        if fit3.success:
            fit, meas, acc = fit3, m3, ok3

    inlier = np.zeros(len(idx), bool)
    inlier[np.flatnonzero(acc)[fit.inliers]] = True
    uv_fit = camera.project((t.X[idx] - fit.C) @ fit.R.T)
    uv_apr = vis["uv"][use]
    sel = np.flatnonzero(use)
    matches = pd.DataFrame({
        "landmark_id": t.landmark_id[idx],
        "view_image": t.view_image[idx, view],
        "sun_angle_deg": sun_ang[use],
        "u_apriori": uv_apr[:, 0], "v_apriori": uv_apr[:, 1],
        "u": meas["u"], "v": meas["v"],
        "ncc": meas["ncc"], "ncc_second": meas["ncc_second"],
        "accepted": acc, "inlier": inlier,
        "residual_u_px": uv_fit[:, 0] - meas["u"], "residual_v_px": uv_fit[:, 1] - meas["v"],
        "emission_deg": vis["emission_deg"][sel], "incidence_deg": vis["incidence_deg"][sel],
        "range_km": vis["range_km"][sel], "template_px": T,
        "valid_fraction": meas["valid_fraction"],
    })
    info.update(R=fit.R, C=fit.C, fit=fit, num_accepted=int(acc.sum()))
    return matches, info


def _pose_row(name: str, row, info: dict, matches: pd.DataFrame, camera: FramingCamera,
              opt: MatchOptions, ref: tuple | None, X_inl: np.ndarray) -> dict:
    """One ``poses.csv`` row from the output of :func:`match_image`."""
    fit: PoseFit = info["fit"]
    du, dv, tw = pointing_offsets(info["R"], info["R0"], camera)
    q = matrix_to_quat(info["R"])
    inl = matches[matches.inlier]
    rec = {
        "image": name, "sequence": row.get("sequence"), "utc": row.get("utc"),
        "range_km": float(np.linalg.norm(info["C0"])), "gsd_km": info["gsd_km"],
        "phase_deg": row.get("phase_deg"), "template_px": info["template_px"],
        "num_visible": info["num_visible"], "num_searched": info["num_searched"],
        "num_accepted": info["num_accepted"], "num_inliers": int(matches.inlier.sum()),
        "coarse_matches": info["coarse_matches"],
        "coarse_shift_u_px": info["coarse_shift_px"][0], "coarse_shift_v_px": info["coarse_shift_px"][1],
        "median_ncc": float(inl.ncc.median()) if len(inl) else np.nan,
        "prefit_rms_px": fit.prefit_rms_px, "postfit_rms_px": fit.postfit_rms_px,
        "du_px": du, "dv_px": dv, "twist_deg": tw,
        "correction_arcsec": rotation_angle_deg(info["R"], info["R0"]) * 3600,
        "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
        "x_km": info["C"][0], "y_km": info["C"][1], "z_km": info["C"][2],
        "sigma_rx_arcsec": fit.sigma_rot_arcsec[0], "sigma_ry_arcsec": fit.sigma_rot_arcsec[1],
        "sigma_rz_arcsec": fit.sigma_rot_arcsec[2],
        "position_estimated": opt.estimate_position, "success": fit.success,
    }
    if opt.estimate_position:
        rec.update(sigma_x_km=fit.sigma_pos_km[0], sigma_y_km=fit.sigma_pos_km[1],
                   sigma_z_km=fit.sigma_pos_km[2])
    if ref is not None and len(X_inl):
        R_s, C_s = ref
        uv_ref = camera.project((X_inl - C_s) @ R_s.T)
        uv_est = camera.project((X_inl - info["C"]) @ info["R"].T)
        uv_apr = camera.project((X_inl - info["C0"]) @ info["R0"].T)
        rec.update(
            sfm_measurement_rms_px=_rms(inl[["u", "v"]].to_numpy() - uv_ref),
            sfm_estimate_rms_px=_rms(uv_est - uv_ref),
            sfm_apriori_rms_px=_rms(uv_apr - uv_ref))
    return rec


def match_images(meta: pd.DataFrame, t: Templates, body: Body, camera: FramingCamera,
                 opt: MatchOptions | None = None, flip: str | None = None,
                 catalog_cameras: pd.DataFrame | None = None, progress: bool = True):
    """Run :func:`match_image` on every image of a metadata table.

    Parameters
    ----------
    meta : pandas.DataFrame
        Metadata of the new images (:func:`~asteroid_colmap.metadata.build_metadata`).
    t : Templates
        Maplets.
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    opt : MatchOptions, optional
        Settings.
    flip : str, optional
        Image flip; defaults to the one the templates were built with.
    catalog_cameras : pandas.DataFrame, optional
        The catalog's ``cameras.csv``. Images found there are matched without their own
        reference views, and their results are compared with the catalog pose
        (``sfm_*_rms_px`` columns).
    progress : bool
        Show a progress bar.

    Returns
    -------
    matches : pandas.DataFrame
        :func:`match_image` rows of all images, with ``image``, the pixel frames of the
        measurement (:func:`~asteroid_colmap.catalog.pixel_frames`) and the catalog
        coordinates of the landmark.
    poses : pandas.DataFrame
        One row per image: counts, residuals, the pointing correction (``du_px``, ``dv_px``,
        ``twist_deg``, ``correction_arcsec``), the estimated attitude ``qw..qz``
        (body-fixed -> camera) and position, and their 1-sigma.
    """
    opt = opt or MatchOptions()
    flip = flip or t.flip
    refs = {}
    if catalog_cameras is not None and "sfm_qw" in catalog_cameras.columns:
        for r in catalog_cameras[catalog_cameras.registered.astype(bool)].itertuples(index=False):
            refs[r.image] = (quat_to_matrix([r.sfm_qw, r.sfm_qx, r.sfm_qy, r.sfm_qz]),
                             np.array([r.sfm_x_km, r.sfm_y_km, r.sfm_z_km]))
    lid = pd.Series(np.arange(len(t)), index=t.landmark_id)
    all_matches, poses = [], []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="match", unit="image",
                       disable=not progress):
        name = row["image"]
        img = apply_flip(load_fits(row["fit_path"]), flip)
        try:
            mt, info = match_image(img, row, t, body, camera, opt,
                                   exclude_image=name if name in refs else None)
        except ValueError as exc:
            log.warning("%s: %s", name, exc)
            continue
        X_inl = t.X[lid.loc[mt.landmark_id[mt.inlier]].to_numpy()]
        poses.append(_pose_row(name, row, info, mt, camera, opt, refs.get(name), X_inl))
        mt.insert(0, "image", name)
        all_matches.append(mt)
        p = poses[-1]
        log.info("%s: %d/%d matched, %d inliers, offset (%.2f, %.2f) px, RMS %.2f -> %.2f px",
                 name, p["num_accepted"], p["num_searched"], p["num_inliers"], p["du_px"],
                 p["dv_px"], p["prefit_rms_px"], p["postfit_rms_px"])
    matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    if len(matches):
        fr = pixel_frames(matches.u.to_numpy(), matches.v.to_numpy(), flip, camera)
        at = matches.columns.get_loc("v") + 1
        for k, (col, vals) in enumerate(fr.items()):
            matches.insert(at + k, col, vals)
        X = t.X[lid.loc[matches.landmark_id].to_numpy()]
        for k, c in enumerate(("x_km", "y_km", "z_km")):
            matches.insert(2 + k, c, X[:, k])
    return matches, pd.DataFrame(poses)


def summarize_matches(matches: pd.DataFrame, poses: pd.DataFrame, t: Templates,
                      opt: MatchOptions) -> dict:
    """Key numbers of a matching run, for ``summary.json``.

    Parameters
    ----------
    matches, poses : pandas.DataFrame
        Output of :func:`match_images`.
    t : Templates
        Maplets used.
    opt : MatchOptions
        Settings used.

    Returns
    -------
    dict
    """
    ok = poses[poses.success] if len(poses) else poses
    inl = matches[matches.inlier] if len(matches) else matches
    res = np.hypot(inl.residual_u_px, inl.residual_v_px) if len(inl) else pd.Series(dtype=float)
    out = {
        "num_images": int(len(poses)),
        "num_images_with_pose": int(len(ok)),
        "num_templates": int(len(t)),
        "maplet_size": int(t.size),
        "maplet_spacing_km": float(t.spacing_km),
        "num_searched": int(len(matches)),
        "num_accepted": int(matches.accepted.sum()) if len(matches) else 0,
        "num_inliers": int(len(inl)),
        "median_inliers_per_image": float(ok.num_inliers.median()) if len(ok) else 0.0,
        "median_ncc": float(inl.ncc.median()) if len(inl) else float("nan"),
        "median_residual_px": float(res.median()) if len(res) else float("nan"),
        "median_prefit_rms_px": float(ok.prefit_rms_px.median()) if len(ok) else float("nan"),
        "median_postfit_rms_px": float(ok.postfit_rms_px.median()) if len(ok) else float("nan"),
        "median_correction_px": float(np.hypot(ok.du_px, ok.dv_px).median()) if len(ok) else float("nan"),
        "median_correction_arcsec": float(ok.correction_arcsec.median()) if len(ok) else float("nan"),
        "options": asdict(opt),
    }
    for k in ("sfm_measurement_rms_px", "sfm_estimate_rms_px", "sfm_apriori_rms_px"):
        if k in ok.columns and ok[k].notna().any():
            out["median_" + k] = float(ok[k].median())
    return out
