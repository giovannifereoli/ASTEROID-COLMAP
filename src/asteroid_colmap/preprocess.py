"""FITS -> COLMAP-ready 8-bit PNGs and feature masks, with an image-orientation check.

COLMAP's camera frame is x right / y down (row index) / z forward. For the label attitude
(J2000 -> DAWN_FC2) to be directly comparable with COLMAP's poses - and for the model not to
come out mirrored - the PNG rows/columns must follow the IK sample/line axes. Rather than
trusting the display conventions, :func:`detect_flip` renders the lit reference ellipsoid
from each label's geometry and picks the array flip whose body silhouette matches best.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from .camera import FramingCamera
from .config import Body
from .geometry import ray_ellipsoid
from .metadata import label_geometry

log = logging.getLogger(__name__)

FLIPS = ("none", "ud", "lr", "rot180")


def load_fits(path: str | Path) -> np.ndarray:
    """First 2-D image HDU of a FITS file, in stored row order.

    Parameters
    ----------
    path : str or pathlib.Path
        FITS file.

    Returns
    -------
    numpy.ndarray
        ``float32`` image, NaN set to 0 and infinities clipped to finite values
        (:func:`numpy.nan_to_num`).

    Raises
    ------
    ValueError
        If the file has no 2-D HDU.
    """
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                return np.nan_to_num(np.asarray(hdu.data, dtype=np.float32))
    raise ValueError(f"no 2-D image in {path}")


def apply_flip(img: np.ndarray, flip: str) -> np.ndarray:
    """Flip an image array.

    Parameters
    ----------
    img : numpy.ndarray
        2-D image.
    flip : str
        One of :data:`FLIPS`: ``"none"``, ``"ud"`` (rows reversed), ``"lr"`` (columns
        reversed) or ``"rot180"`` (both).

    Returns
    -------
    numpy.ndarray
        A flipped view of ``img`` (no copy).

    Raises
    ------
    ValueError
        If ``flip`` is not in :data:`FLIPS`.
    """
    if flip not in FLIPS:
        raise ValueError(f"flip must be one of {FLIPS}")
    return {"none": img, "ud": img[::-1], "lr": img[:, ::-1], "rot180": img[::-1, ::-1]}[flip]


def body_mask(img: np.ndarray, rel_threshold: float = 0.06) -> np.ndarray:
    """Mask of the illuminated body.

    Keeps the pixels brighter than ``rel_threshold`` times the 99.5th percentile, then
    the largest connected component, with its holes filled.

    Parameters
    ----------
    img : numpy.ndarray
        2-D image.
    rel_threshold : float
        Threshold as a fraction of the 99.5th percentile.

    Returns
    -------
    numpy.ndarray
        Boolean mask, same shape as ``img``.
    """
    bright = np.percentile(img, 99.5)
    mask = img > rel_threshold * bright
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(mask)


def feature_mask(body: np.ndarray, erode_px: float = 8.0) -> np.ndarray:
    """Body mask shrunk by ``erode_px`` away from the limb and terminator.

    Where the body touches the frame edge the mask is not eroded, because the distance
    transform does not treat the edge as background.

    Parameters
    ----------
    body : numpy.ndarray
        Boolean body mask (:func:`body_mask`).
    erode_px : float
        Minimum distance to the nearest background pixel (px); ``<= 0`` disables the
        erosion.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the pixels where COLMAP may detect features.
    """
    if erode_px <= 0:
        return body
    return ndimage.distance_transform_edt(body) > erode_px


def to_uint8(img: np.ndarray, body: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Linear stretch to 8 bits, from the sky level to the 99.8th percentile of the body.

    Parameters
    ----------
    img : numpy.ndarray
        2-D image.
    body : numpy.ndarray
        Boolean body mask; the sky level is the median outside it (0 if there is no sky).
    gamma : float
        Exponent applied after the stretch to [0, 1]; 1 keeps it linear.

    Returns
    -------
    numpy.ndarray
        ``uint8`` image.
    """
    lo = float(np.median(img[~body])) if (~body).any() else 0.0
    hi = float(np.percentile(img[body], 99.8)) if body.any() else float(img.max())
    out = np.clip((img - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if gamma != 1.0:
        out = out**gamma
    return np.round(out * 255).astype(np.uint8)


def sample_grid(camera: FramingCamera, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of a decimated pixel grid, starting at ``step // 2``.

    Parameters
    ----------
    camera : FramingCamera
        Supplies the image size.
    step : int
        Decimation factor (px).

    Returns
    -------
    rows, cols : numpy.ndarray
        Integer indices.
    """
    return np.arange(step // 2, camera.height, step), np.arange(step // 2, camera.width, step)


def predicted_lit_mask(row, body: Body, camera: FramingCamera, step: int = 4) -> np.ndarray:
    """Lit part of the reference ellipsoid as seen with the label geometry.

    Rays through the centres of the :func:`sample_grid` pixels are intersected with the
    ellipsoid; a pixel counts as lit if its ray hits and the outward surface normal
    faces the Sun. Pixel axes are COLMAP's (x right, y down), i.e. the IK sample/line
    axes.

    Parameters
    ----------
    row : pandas.Series
        Metadata row (:func:`~asteroid_colmap.metadata.label_geometry` inputs).
    body : Body
        Target body (rotation model and ellipsoid).
    camera : FramingCamera
        Camera model.
    step : int
        Grid decimation (px).

    Returns
    -------
    numpy.ndarray
        Boolean mask on the decimated grid, ``(len(rows), len(cols))``.
    """
    rows, cols = sample_grid(camera, step)
    U, V = np.meshgrid(cols + 0.5, rows + 0.5)
    g = label_geometry(row, body)
    rays_bf = camera.pixel_rays(U, V) @ g["R_cam_from_bf"]  # row-vector form of R^T x
    pts, hit = ray_ellipsoid(g["sc_bf"], rays_bf, body.radii_km)
    normals = np.nan_to_num(pts) / np.square(body.radii_km)
    return hit & (normals @ g["sun_bf"] > 0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks; 0 when both are empty."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def orientation_scores(
    meta: pd.DataFrame, body: Body, camera: FramingCamera, n_samples: int = 12, step: int = 4,
    rel_threshold: float = 0.06,
) -> pd.DataFrame:
    """Silhouette agreement for every candidate flip on a sample of images.

    For ``n_samples`` images spread evenly over ``meta``, the observed body mask of each
    flipped FITS array is compared with :func:`predicted_lit_mask`.

    Parameters
    ----------
    meta : pandas.DataFrame
        Image metadata, with ``fit_path``.
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    n_samples : int
        Number of images to test.
    step : int
        Grid decimation (px) for the comparison.
    rel_threshold : float
        Passed to :func:`body_mask`.

    Returns
    -------
    pandas.DataFrame
        One row per image and flip: ``image``, ``flip`` and ``iou``.
    """
    idx = np.unique(np.linspace(0, len(meta) - 1, min(n_samples, len(meta))).round().astype(int))
    rows, cols = sample_grid(camera, step)
    records = []
    for i in idx:
        r = meta.iloc[i]
        pred = predicted_lit_mask(r, body, camera, step)
        img = load_fits(r["fit_path"])
        for flip in FLIPS:
            obs = body_mask(apply_flip(img, flip), rel_threshold)[np.ix_(rows, cols)]
            records.append({"image": r["image"], "flip": flip, "iou": _iou(obs, pred)})
    return pd.DataFrame(records)


def detect_flip(meta, body, camera, **kw) -> tuple[str, pd.DataFrame]:
    """Choose the array flip whose silhouettes best match the label geometry.

    Logs the mean IoU of every flip, and warns if the best mean is below 0.5 or beats
    the runner-up by less than 0.03.

    Parameters
    ----------
    meta, body, camera
        See :func:`orientation_scores`.
    **kw
        Passed to :func:`orientation_scores`.

    Returns
    -------
    flip : str
        The flip with the highest mean IoU.
    scores : pandas.DataFrame
        Output of :func:`orientation_scores`.
    """
    scores = orientation_scores(meta, body, camera, **kw)
    mean = scores.groupby("flip")["iou"].mean().sort_values(ascending=False)
    best = str(mean.index[0])
    log.info("orientation check (mean IoU): %s", ", ".join(f"{k}={v:.3f}" for k, v in mean.items()))
    if mean.iloc[0] < 0.5 or mean.iloc[0] - mean.iloc[1] < 0.03:
        log.warning("orientation check is ambiguous - inspect plots/orientation_check.png")
    return best, scores


def prepare(
    meta: pd.DataFrame,
    images_dir: Path,
    masks_dir: Path,
    body: Body,
    camera: FramingCamera,
    flip: str = "auto",
    erode_px: float = 8.0,
    rel_threshold: float = 0.06,
    gamma: float = 1.0,
) -> dict:
    """Write the COLMAP input images and feature masks.

    Each FITS image is flipped, stretched to 8 bits and written as
    ``images_dir/<image>``; its feature mask is written as ``masks_dir/<image>.png``,
    the name COLMAP expects for ``ImageReader.mask_path``.

    Parameters
    ----------
    meta : pandas.DataFrame
        Image metadata, with ``image`` and ``fit_path``.
    images_dir, masks_dir : pathlib.Path
        Output directories; created if needed.
    body : Body
        Target body, for the orientation check.
    camera : FramingCamera
        Camera model; every image must match its size.
    flip : str
        One of :data:`FLIPS`, or ``"auto"`` to run :func:`detect_flip`.
    erode_px : float
        Mask erosion (px), see :func:`feature_mask`.
    rel_threshold : float
        Body threshold, see :func:`body_mask`.
    gamma : float
        Stretch exponent, see :func:`to_uint8`.

    Returns
    -------
    dict
        ``flip`` (the one applied), ``erode_px``, ``rel_threshold``, ``gamma``,
        ``orientation_scores`` (list of records, or ``None`` if ``flip`` was given) and
        ``images`` (per image: ``image``, ``body_fraction``, ``feature_fraction``).

    Raises
    ------
    ValueError
        If an image does not have the camera's size.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    scores = None
    if flip == "auto":
        flip, scores = detect_flip(meta, body, camera, rel_threshold=rel_threshold)
    stats = []
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="prepare", unit="img"):
        img = apply_flip(load_fits(r["fit_path"]), flip)
        if img.shape != (camera.height, camera.width):
            raise ValueError(f"{r['image']}: shape {img.shape} does not match the camera model")
        body_px = body_mask(img, rel_threshold)
        keep = feature_mask(body_px, erode_px)
        Image.fromarray(to_uint8(img, body_px, gamma)).save(images_dir / r["image"])
        Image.fromarray((keep * 255).astype(np.uint8)).save(masks_dir / f"{r['image']}.png")
        stats.append({"image": r["image"], "body_fraction": float(body_px.mean()),
                      "feature_fraction": float(keep.mean())})
    return {
        "flip": flip,
        "erode_px": erode_px,
        "rel_threshold": rel_threshold,
        "gamma": gamma,
        "orientation_scores": None if scores is None else scores.to_dict(orient="records"),
        "images": stats,
    }
