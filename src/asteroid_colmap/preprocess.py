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
    """First 2-D image HDU as float32, NaNs set to 0, in stored row order."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                return np.nan_to_num(np.asarray(hdu.data, dtype=np.float32))
    raise ValueError(f"no 2-D image in {path}")


def apply_flip(img: np.ndarray, flip: str) -> np.ndarray:
    if flip not in FLIPS:
        raise ValueError(f"flip must be one of {FLIPS}")
    return {"none": img, "ud": img[::-1], "lr": img[:, ::-1], "rot180": img[::-1, ::-1]}[flip]


def body_mask(img: np.ndarray, rel_threshold: float = 0.06) -> np.ndarray:
    """Illuminated body: pixels brighter than ``rel_threshold`` x the 99.5th percentile,
    largest connected component, holes filled."""
    bright = np.percentile(img, 99.5)
    mask = img > rel_threshold * bright
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(mask)


def feature_mask(body: np.ndarray, erode_px: float = 8.0) -> np.ndarray:
    """Body mask shrunk by ``erode_px`` away from the limb/terminator (frame edges kept)."""
    if erode_px <= 0:
        return body
    return ndimage.distance_transform_edt(body) > erode_px


def to_uint8(img: np.ndarray, body: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Linear stretch from the sky level to the 99.8th percentile of the body."""
    lo = float(np.median(img[~body])) if (~body).any() else 0.0
    hi = float(np.percentile(img[body], 99.8)) if body.any() else float(img.max())
    out = np.clip((img - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if gamma != 1.0:
        out = out**gamma
    return np.round(out * 255).astype(np.uint8)


def sample_grid(camera: FramingCamera, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of a decimated pixel grid."""
    return np.arange(step // 2, camera.height, step), np.arange(step // 2, camera.width, step)


def predicted_lit_mask(row, body: Body, camera: FramingCamera, step: int = 4) -> np.ndarray:
    """Lit part of the reference ellipsoid as seen with the label geometry (COLMAP pixel axes)."""
    rows, cols = sample_grid(camera, step)
    U, V = np.meshgrid(cols + 0.5, rows + 0.5)
    g = label_geometry(row, body)
    rays_bf = camera.pixel_rays(U, V) @ g["R_cam_from_bf"]  # row-vector form of R^T x
    pts, hit = ray_ellipsoid(g["sc_bf"], rays_bf, body.radii_km)
    normals = np.nan_to_num(pts) / np.square(body.radii_km)
    return hit & (normals @ g["sun_bf"] > 0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def orientation_scores(
    meta: pd.DataFrame, body: Body, camera: FramingCamera, n_samples: int = 12, step: int = 4,
    rel_threshold: float = 0.06,
) -> pd.DataFrame:
    """IoU between observed and predicted silhouettes for every candidate flip."""
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
    """Write ``images_dir/<name>.png`` and COLMAP masks ``masks_dir/<name>.png.png``."""
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
