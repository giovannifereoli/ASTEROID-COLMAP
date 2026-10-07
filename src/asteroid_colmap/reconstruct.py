"""COLMAP sparse reconstruction driven through the ``colmap`` command-line interface.

The intrinsics come from the instrument kernel and stay fixed; features are restricted to
the eroded body mask; image pairs come from the label geometry (see :mod:`.pairs`).

Narrow-angle pitfall: COLMAP treats a camera whose focal length exceeds
``Mapper.max_focal_length_ratio`` (default 10) times the image size as bogus and silently
triangulates nothing from it - the mapper then reports "No good initial image pair found"
and the triangulator returns an empty model. The Dawn FC ratio is 10.5, so every mapping
call raises the limit (:func:`_camera_limits`).

Two mapping modes:

``label-poses`` (default)
    The PDS label poses seed the model: COLMAP ``point_triangulator`` builds tracks with the
    poses held fixed, ``bundle_adjuster`` then refines poses and points, and the pair is
    repeated. The model comes out in the body-fixed frame and km, needs no initial pair
    (with a 5.5 deg field of view the views are nearly affine and two-view geometry is
    weakly conditioned), and is 8x faster than incremental mapping on the RC3 set.
``incremental``
    COLMAP's standard ``mapper`` from scratch, no prior poses; the model is tied to the
    body frame afterwards by the similarity fit in :mod:`.georef`. On the RC3 set it
    registers all images and agrees with the label-poses catalog to 15 m after removing
    a common offset, so it serves as an independent check of the label pointing.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .camera import FramingCamera
from .config import Body
from .geometry import matrix_to_quat
from .metadata import label_geometry
from .model_io import count_registered_images
from .pairs import view_angle_pairs, write_pairs
from .workspace import Workspace

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "COLMAP executable not found. Install it (macOS: `brew install colmap`; conda: "
    "`conda install -c conda-forge colmap`; Ubuntu: `apt install colmap`; or build from "
    "https://github.com/colmap/colmap), or point $COLMAP_BIN / --colmap at the binary."
)


MODES = ("label-poses", "incremental")


@dataclass
class ReconstructionOptions:
    mode: str = "label-poses"  # label-poses | incremental (see module docstring)
    refine_poses: bool = True  # label-poses: bundle-adjust the poses, not only the points
    rounds: int = 2  # label-poses: triangulate + bundle-adjust iterations
    matching: str = "pairs"  # pairs | exhaustive | sequential
    max_view_angle_deg: float = 40.0  # pairs: max angle between sub-spacecraft directions
    sequential_overlap: int = 10
    max_num_features: int = 8192
    peak_threshold: float = 0.004  # below SIFT's 0.0067 default: low-contrast regolith
    affine_shape: bool = False
    domain_size_pooling: bool = True
    max_ratio: float = 0.8
    min_num_inliers: int = 15
    max_error_px: float = 4.0
    guided_matching: bool = True
    init_min_tri_angle_deg: float = 8.0  # consecutive frames are ~5.6 deg apart
    init_min_num_inliers: int = 100
    abs_pose_min_num_inliers: int = 30
    filter_max_reproj_error_px: float = 3.0
    tri_min_angle_deg: float = 1.5
    min_model_size: int = 10
    num_threads: int = -1
    use_gpu: bool = False


def colmap_executable(explicit: str | None = None) -> str:
    exe = explicit or os.environ.get("COLMAP_BIN") or shutil.which("colmap")
    if not exe or not (Path(exe).exists() or shutil.which(exe)):
        raise FileNotFoundError(INSTALL_HINT)
    return exe


def colmap_version(exe: str) -> str:
    out = subprocess.run([exe, "help"], capture_output=True, text=True).stdout
    return next((ln.strip() for ln in out.splitlines() if ln.startswith("COLMAP")), "unknown")


def run_colmap(exe: str, command: str, options: dict, log_dir: Path, log_name: str | None = None) -> None:
    """Run ``colmap <command> --key value ...``; output goes to ``log_dir/<log_name>.log``."""
    args = [exe, command]
    for k, v in options.items():
        if isinstance(v, bool):
            v = int(v)
        args += [f"--{k}", str(v)]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name or command}.log"
    log.info("colmap %s (log: %s)", command, log_path)
    t0 = time.time()
    with open(log_path, "w") as fh:
        fh.write(" ".join(args) + "\n\n")
        fh.flush()
        proc = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-25:])
        raise RuntimeError(f"colmap {command} failed (exit {proc.returncode}):\n{tail}")
    log.info("colmap %s finished in %.0f s", command, time.time() - t0)


def extract_features(exe, ws: Workspace, camera: FramingCamera, opt: ReconstructionOptions):
    run_colmap(exe, "feature_extractor", {
        "database_path": ws.database,
        "image_path": ws.images,
        "ImageReader.mask_path": ws.masks,
        "ImageReader.camera_model": camera.colmap_model,
        "ImageReader.single_camera": True,
        "ImageReader.camera_params": camera.colmap_params_str(),
        "FeatureExtraction.use_gpu": opt.use_gpu,
        "FeatureExtraction.num_threads": opt.num_threads,
        "SiftExtraction.max_num_features": opt.max_num_features,
        "SiftExtraction.peak_threshold": opt.peak_threshold,
        "SiftExtraction.estimate_affine_shape": opt.affine_shape,
        "SiftExtraction.domain_size_pooling": opt.domain_size_pooling,
    }, ws.logs)


def match_features(exe, ws: Workspace, meta: pd.DataFrame, opt: ReconstructionOptions) -> int:
    common = {
        "database_path": ws.database,
        "FeatureMatching.use_gpu": opt.use_gpu,
        "FeatureMatching.num_threads": opt.num_threads,
        "FeatureMatching.guided_matching": opt.guided_matching,
        "SiftMatching.max_ratio": opt.max_ratio,
        "TwoViewGeometry.min_num_inliers": opt.min_num_inliers,
        "TwoViewGeometry.max_error": opt.max_error_px,
        "TwoViewGeometry.detect_watermark": False,
    }
    if opt.matching == "pairs":
        pairs = view_angle_pairs(meta, opt.max_view_angle_deg)
        write_pairs(pairs, ws.pairs)
        log.info("%d image pairs within %.0f deg of each other", len(pairs), opt.max_view_angle_deg)
        run_colmap(exe, "matches_importer",
                   {**common, "match_list_path": ws.pairs, "match_type": "pairs"}, ws.logs)
        return len(pairs)
    if opt.matching == "exhaustive":
        run_colmap(exe, "exhaustive_matcher", common, ws.logs)
        return len(meta) * (len(meta) - 1) // 2
    if opt.matching == "sequential":
        run_colmap(exe, "sequential_matcher",
                   {**common, "SequentialMatching.overlap": opt.sequential_overlap}, ws.logs)
        return len(meta) * opt.sequential_overlap
    raise ValueError(f"unknown matching mode {opt.matching!r}")


def _fixed_intrinsics(prefix: str) -> dict:
    return {f"{prefix}refine_focal_length": False, f"{prefix}refine_principal_point": False,
            f"{prefix}refine_extra_params": False}


def _camera_limits(camera: FramingCamera) -> dict:
    """Relax COLMAP's "bogus camera" test. The FC focal length is 10.5x the image width, above
    the default limit of 10, and COLMAP then silently refuses to triangulate from the camera."""
    ratio = max(camera.fx, camera.fy) / max(camera.width, camera.height)
    return {"Mapper.max_focal_length_ratio": max(10.0, 2.0 * ratio)}


def database_images(db: Path) -> pd.DataFrame:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
        return pd.read_sql_query("SELECT image_id, name, camera_id FROM images", con)


def write_label_model(ws: Workspace, meta: pd.DataFrame, body: Body, camera: FramingCamera,
                      out: Path) -> int:
    """COLMAP TXT model holding the label poses (body-fixed frame, km) and no points."""
    imgs = database_images(ws.database)
    lookup = meta.set_index("image")
    imgs = imgs[imgs["name"].isin(lookup.index)]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cameras.txt", "w") as fh:
        for cid in sorted(imgs.camera_id.unique()):
            fh.write(f"{cid} {camera.colmap_model} {camera.width} {camera.height} "
                     f"{' '.join(repr(float(v)) for v in camera.colmap_params)}\n")
    with open(out / "images.txt", "w") as fh:
        for r in imgs.itertuples():
            g = label_geometry(lookup.loc[r.name], body)
            R = g["R_cam_from_bf"]
            q = matrix_to_quat(R)
            t = -R @ g["sc_bf"]
            fh.write(" ".join([str(r.image_id), *(repr(float(v)) for v in np.r_[q, t]),
                               str(r.camera_id), r.name]) + "\n\n")
    (out / "points3D.txt").write_text("")
    return len(imgs)


def _triangulate(exe, ws: Workspace, camera: FramingCamera, opt: ReconstructionOptions, src: Path,
                 dst: Path, clear: bool, tag: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    run_colmap(exe, "point_triangulator", {
        "database_path": ws.database,
        "image_path": ws.images,
        "input_path": src,
        "output_path": dst,
        "clear_points": clear,
        "refine_intrinsics": False,
        **_fixed_intrinsics("Mapper.ba_"),
        **_camera_limits(camera),
        "Mapper.num_threads": opt.num_threads,
        "Mapper.filter_max_reproj_error": opt.filter_max_reproj_error_px,
        "Mapper.filter_min_tri_angle": opt.tri_min_angle_deg,
        "Mapper.tri_min_angle": opt.tri_min_angle_deg,
        "Mapper.extract_colors": True,
    }, ws.logs, tag)


def _bundle_adjust(exe, ws: Workspace, opt: ReconstructionOptions, src: Path, dst: Path, tag: str):
    dst.mkdir(parents=True, exist_ok=True)
    run_colmap(exe, "bundle_adjuster", {
        "input_path": src,
        "output_path": dst,
        **_fixed_intrinsics("BundleAdjustment."),
        "BundleAdjustment.refine_rig_from_world": opt.refine_poses,
        "BundleAdjustmentCeres.max_num_iterations": 100,
    }, ws.logs, tag)


def map_with_label_poses(exe, ws: Workspace, meta: pd.DataFrame, body: Body, camera: FramingCamera,
                         opt: ReconstructionOptions) -> Path:
    """label poses -> (point_triangulator -> bundle_adjuster) x rounds; returns the final model."""
    if ws.sparse.exists():
        shutil.rmtree(ws.sparse)
    prior = ws.sparse / "label_poses"
    n = write_label_model(ws, meta, body, camera, prior)
    log.info("seeded %d label poses in %s", n, prior)
    src = prior
    for k in range(1, max(opt.rounds, 1) + 1):
        tri = ws.sparse / f"round{k}_triangulated"
        _triangulate(exe, ws, camera, opt, src, tri, clear=(k == 1), tag=f"point_triangulator_{k}")
        src = tri
        if opt.refine_poses:
            ba = ws.sparse / f"round{k}_adjusted"
            _bundle_adjust(exe, ws, opt, tri, ba, tag=f"bundle_adjuster_{k}")
            src = ba
    return src


def map_images(exe, ws: Workspace, camera: FramingCamera, opt: ReconstructionOptions) -> None:
    if ws.sparse.exists():
        shutil.rmtree(ws.sparse)
    ws.sparse.mkdir(parents=True)
    run_colmap(exe, "mapper", {
        "database_path": ws.database,
        "image_path": ws.images,
        "output_path": ws.sparse,
        "Mapper.num_threads": opt.num_threads,
        **_fixed_intrinsics("Mapper.ba_"),
        **_camera_limits(camera),
        "Mapper.init_min_tri_angle": opt.init_min_tri_angle_deg,
        "Mapper.init_min_num_inliers": opt.init_min_num_inliers,
        "Mapper.abs_pose_min_num_inliers": opt.abs_pose_min_num_inliers,
        "Mapper.filter_max_reproj_error": opt.filter_max_reproj_error_px,
        "Mapper.filter_min_tri_angle": opt.tri_min_angle_deg,
        "Mapper.tri_min_angle": opt.tri_min_angle_deg,
        "Mapper.multiple_models": True,
        "Mapper.min_model_size": opt.min_model_size,
        "Mapper.extract_colors": True,
    }, ws.logs)


def export_models(exe, ws: Workspace, subdirs: list[Path] | None = None) -> list[dict]:
    """Convert sub-models to TXT; returns them sorted by registered images (desc)."""
    if ws.model_txt.exists():
        shutil.rmtree(ws.model_txt)
    models = []
    subdirs = subdirs if subdirs is not None else sorted(p for p in ws.sparse.iterdir() if p.is_dir())
    for sub in subdirs:
        out = ws.model_txt / sub.name
        out.mkdir(parents=True)
        run_colmap(exe, "model_converter",
                   {"input_path": sub, "output_path": out, "output_type": "TXT"}, ws.logs,
                   f"model_converter_{sub.name}")
        models.append({"name": sub.name, "path": str(out),
                       "num_images": count_registered_images(out)})
    return sorted(models, key=lambda m: -m["num_images"])


def reconstruct(
    ws: Workspace,
    meta: pd.DataFrame,
    body: Body,
    camera: FramingCamera,
    opt: ReconstructionOptions | None = None,
    colmap: str | None = None,
    overwrite: bool = False,
    skip_features: bool = False,
) -> dict:
    opt = opt or ReconstructionOptions()
    if opt.mode not in MODES:
        raise ValueError(f"unknown mode {opt.mode!r}; choose from {MODES}")
    exe = colmap_executable(colmap)
    if not any(ws.images.glob("*.png")):
        raise FileNotFoundError(f"no images in {ws.images}; run the prepare step first")
    ws.colmap.mkdir(parents=True, exist_ok=True)
    timings = {}
    if not skip_features:
        if ws.database.exists():
            if not overwrite:
                raise FileExistsError(f"{ws.database} exists; pass --overwrite to start over "
                                      "or --skip-features to re-run only the mapper")
            for p in ws.colmap.glob("database.db*"):
                p.unlink()
        t0 = time.time()
        extract_features(exe, ws, camera, opt)
        timings["features_s"] = time.time() - t0
        t0 = time.time()
        n_pairs = match_features(exe, ws, meta, opt)
        timings["matching_s"] = time.time() - t0
    else:
        n_pairs = None
    t0 = time.time()
    if opt.mode == "label-poses":
        models = export_models(exe, ws, [map_with_label_poses(exe, ws, meta, body, camera, opt)])
    else:
        try:
            map_images(exe, ws, camera, opt)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}\nThe incremental mapper failed; see colmap/logs/mapper.log. "
                               "If it found no initial pair, try --init-min-tri-angle 4 or "
                               "--mode label-poses") from None
        models = export_models(exe, ws)
    timings["mapping_s"] = time.time() - t0
    if not models:
        raise RuntimeError("COLMAP produced no model; see colmap/logs/")
    info = {
        "colmap": colmap_version(exe),
        "options": asdict(opt),
        "camera": {"model": camera.colmap_model, "params": camera.colmap_params},
        "num_input_images": len(meta),
        "num_pairs": n_pairs,
        "models": models,
        "selected_model": models[0]["path"],
        "timings": timings,
    }
    ws.write_json(ws.reconstruct_json, info)
    log.info("selected model %s with %d/%d images", models[0]["name"], models[0]["num_images"],
             len(meta))
    return info
