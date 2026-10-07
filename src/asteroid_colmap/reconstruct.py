"""COLMAP sparse reconstruction driven through the ``colmap`` command-line interface.

The intrinsics come from the instrument kernel and stay fixed; features are restricted to
the eroded body mask; image pairs come from the label geometry (see :mod:`.pairs`).

Narrow-angle pitfall: COLMAP treats a camera whose focal length exceeds
``Mapper.max_focal_length_ratio`` (default 10) times the image size as bogus and silently
triangulates nothing from it - the mapper then reports "No good initial image pair found"
and the triangulator returns an empty model. The Dawn FC ratio is 10.5, so every mapping
call raises the limit (:func:`_camera_limits`).

Orientation pitfall: SIFT stores a keypoint with two dominant orientations as two keypoints
at the same pixel, and the copies can join different tracks, giving two landmarks for one
surface feature. COLMAP's covariant extractor, used for DSP-SIFT, ignores
``SiftExtraction.max_num_orientations``, so the copies are removed from the database after
extraction (:func:`drop_duplicate_keypoints`).

Two mapping modes:

``label-poses`` (default)
    The PDS label poses seed the model: COLMAP ``point_triangulator`` builds tracks with the
    poses held fixed, ``bundle_adjuster`` then refines poses and points, and the pair is
    repeated. The model comes out in the body-fixed frame and km, needs no initial pair
    (with a 5.5 deg field of view the views are nearly affine and two-view geometry is
    weakly conditioned), and is about 7x faster than incremental mapping on the RC3 set.
``incremental``
    COLMAP's standard ``mapper`` from scratch, no prior poses; the model is tied to the
    body frame afterwards by the similarity fit in :mod:`.georef`. On the RC3 set it
    registers all images and agrees with the label-poses catalog to 16 m after removing
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
    """Settings of the COLMAP feature, matching and mapping steps.

    Attributes
    ----------
    mode : str
        Mapping mode, one of :data:`MODES` (see the module docstring).
    refine_poses : bool
        ``label-poses``: bundle-adjust the poses as well as the points.
    rounds : int
        ``label-poses``: number of triangulate + bundle-adjust rounds.
    matching : str
        ``"pairs"`` (view-angle pairs, :mod:`.pairs`), ``"exhaustive"`` or
        ``"sequential"``.
    max_view_angle_deg : float
        ``pairs``: maximum angle between the sub-spacecraft directions of a pair (deg).
    sequential_overlap : int
        ``sequential``: number of following images each image is matched with.
    max_num_features : int
        Maximum number of SIFT features per image.
    peak_threshold : float
        SIFT peak threshold; below COLMAP's 0.0067 default because the regolith has low
        contrast.
    max_num_orientations : int
        Orientations per SIFT keypoint. A second orientation is stored as a second keypoint
        at the same pixel; the two copies can join different tracks and turn one surface
        feature into two landmarks, so 1 is used. COLMAP's covariant extractor (used with
        ``domain_size_pooling`` or ``affine_shape``) ignores the option, so with 1 the extra
        copies are also removed from the database (:func:`drop_duplicate_keypoints`).
    affine_shape : bool
        Estimate affine feature shapes.
    domain_size_pooling : bool
        Use DSP-SIFT descriptors.
    max_ratio : float
        Lowe ratio-test threshold.
    min_num_inliers : int
        Minimum number of two-view inliers for a verified pair.
    max_error_px : float
        Maximum epipolar error in two-view verification (px).
    guided_matching : bool
        Re-match under the estimated two-view geometry.
    init_min_tri_angle_deg : float
        ``incremental``: minimum triangulation angle of the initial pair (deg);
        consecutive frames are about 5.6 deg apart.
    init_min_num_inliers : int
        ``incremental``: minimum number of inliers of the initial pair.
    abs_pose_min_num_inliers : int
        ``incremental``: minimum number of 2-D/3-D inliers to register an image.
    filter_max_reproj_error_px : float
        Observations with a larger reprojection error are removed (px).
    tri_min_angle_deg : float
        Minimum triangulation angle to create or keep a point (deg).
    min_model_size : int
        ``incremental``: minimum number of images in a kept sub-model.
    num_threads : int
        COLMAP threads; -1 uses all cores.
    use_gpu : bool
        Run SIFT extraction and matching on the GPU.
    """
    mode: str = "label-poses"  # label-poses | incremental (see module docstring)
    refine_poses: bool = True  # label-poses: bundle-adjust the poses, not only the points
    rounds: int = 2  # label-poses: triangulate + bundle-adjust iterations
    matching: str = "pairs"  # pairs | exhaustive | sequential
    max_view_angle_deg: float = 40.0  # pairs: max angle between sub-spacecraft directions
    sequential_overlap: int = 10
    max_num_features: int = 8192
    peak_threshold: float = 0.004  # below SIFT's 0.0067 default: low-contrast regolith
    max_num_orientations: int = 1  # more duplicate keypoints, hence landmarks
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
    """Locate the COLMAP binary.

    Parameters
    ----------
    explicit : str, optional
        Path or command name; otherwise ``$COLMAP_BIN``, then ``colmap`` on the ``PATH``.

    Returns
    -------
    str
        Executable to call.

    Raises
    ------
    FileNotFoundError
        If no executable is found; the message is :data:`INSTALL_HINT`.
    """
    exe = explicit or os.environ.get("COLMAP_BIN") or shutil.which("colmap")
    if not exe or not (Path(exe).exists() or shutil.which(exe)):
        raise FileNotFoundError(INSTALL_HINT)
    return exe


def colmap_version(exe: str) -> str:
    """Version line printed by ``colmap help``.

    Parameters
    ----------
    exe : str
        COLMAP executable.

    Returns
    -------
    str
        First output line that starts with ``COLMAP``, or ``"unknown"``.
    """
    out = subprocess.run([exe, "help"], capture_output=True, text=True).stdout
    return next((ln.strip() for ln in out.splitlines() if ln.startswith("COLMAP")), "unknown")


def run_colmap(exe: str, command: str, options: dict, log_dir: Path, log_name: str | None = None) -> None:
    """Run ``colmap <command> --key value ...`` and log its output.

    Parameters
    ----------
    exe : str
        COLMAP executable.
    command : str
        COLMAP sub-command, e.g. ``"feature_extractor"``.
    options : dict
        Command-line options without the leading ``--``; booleans are passed as 0/1 and
        other values with ``str``.
    log_dir : pathlib.Path
        Directory of the log file; created if needed.
    log_name : str, optional
        Log file stem; default ``command``. The log starts with the full command line.

    Raises
    ------
    RuntimeError
        If COLMAP exits with a non-zero status; the message ends with the last 25 log
        lines.
    """
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
    """Run ``feature_extractor`` on the prepared images, with their masks.

    All images share one camera with the fixed IK intrinsics. With
    ``opt.max_num_orientations == 1``, the same-pixel copies left by the covariant
    extractor are then removed.

    Parameters
    ----------
    exe : str
        COLMAP executable.
    ws : Workspace
        Working directory (images, masks, database, logs).
    camera : FramingCamera
        Camera model.
    opt : ReconstructionOptions
        SIFT and threading options.
    """
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
        "SiftExtraction.max_num_orientations": opt.max_num_orientations,
        "SiftExtraction.estimate_affine_shape": opt.affine_shape,
        "SiftExtraction.domain_size_pooling": opt.domain_size_pooling,
    }, ws.logs)
    if opt.max_num_orientations == 1:
        before, after = drop_duplicate_keypoints(ws.database)
        log.info("kept %d of %d keypoints (one per pixel)", after, before)


def drop_duplicate_keypoints(db: Path) -> tuple[int, int]:
    """Keep one keypoint per pixel in every image of a COLMAP database.

    COLMAP's covariant SIFT extractor ignores ``SiftExtraction.max_num_orientations`` and
    stores a keypoint once per orientation, every copy at the same pixel. This keeps the
    first copy and its descriptor. Run it after feature extraction and before matching.

    Parameters
    ----------
    db : pathlib.Path
        ``database.db``, modified in place.

    Returns
    -------
    tuple of int
        Number of keypoints before and after.
    """
    before = after = 0
    with closing(sqlite3.connect(db)) as con, con:
        rows = con.execute("SELECT k.image_id, k.rows, k.cols, k.data, d.data "
                           "FROM keypoints k JOIN descriptors d USING (image_id)").fetchall()
        for image_id, n, cols, kdata, ddata in rows:
            before += n
            if n == 0:
                continue
            kp = np.frombuffer(kdata, np.float32).reshape(n, cols)
            keep = np.sort(np.unique(kp[:, :2], axis=0, return_index=True)[1])
            after += len(keep)
            if len(keep) < n:
                desc = np.frombuffer(ddata, np.uint8).reshape(n, -1)  # raw bytes: any dtype
                con.execute("UPDATE keypoints SET rows = ?, data = ? WHERE image_id = ?",
                            (len(keep), kp[keep].tobytes(), image_id))
                con.execute("UPDATE descriptors SET rows = ?, data = ? WHERE image_id = ?",
                            (len(keep), desc[keep].tobytes(), image_id))
    return before, after


def match_features(exe, ws: Workspace, meta: pd.DataFrame, opt: ReconstructionOptions) -> int:
    """Match features with the selected strategy and verify the pairs geometrically.

    Parameters
    ----------
    exe : str
        COLMAP executable.
    ws : Workspace
        Working directory.
    meta : pandas.DataFrame
        Image metadata; ``pairs`` matching uses its label geometry.
    opt : ReconstructionOptions
        ``matching`` and the matching options.

    Returns
    -------
    int
        Number of image pairs submitted to the matcher (an upper bound for
        ``sequential``).

    Raises
    ------
    ValueError
        If ``opt.matching`` is not ``pairs``, ``exhaustive`` or ``sequential``.
    """
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
    """COLMAP options that freeze focal length, principal point and distortion."""
    return {f"{prefix}refine_focal_length": False, f"{prefix}refine_principal_point": False,
            f"{prefix}refine_extra_params": False}


def _camera_limits(camera: FramingCamera) -> dict:
    """Relax COLMAP's "bogus camera" test.

    The FC focal length is 10.5 times the image width, above COLMAP's default limit of
    10, and COLMAP then silently refuses to triangulate from the camera.
    """
    ratio = max(camera.fx, camera.fy) / max(camera.width, camera.height)
    return {"Mapper.max_focal_length_ratio": max(10.0, 2.0 * ratio)}


def database_images(db: Path) -> pd.DataFrame:
    """Images registered in a COLMAP database.

    Parameters
    ----------
    db : pathlib.Path
        ``database.db``, opened read-only.

    Returns
    -------
    pandas.DataFrame
        ``image_id``, ``name`` and ``camera_id``.
    """
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
        return pd.read_sql_query("SELECT image_id, name, camera_id FROM images", con)


def write_label_model(ws: Workspace, meta: pd.DataFrame, body: Body, camera: FramingCamera,
                      out: Path) -> int:
    """Write a COLMAP TXT model holding the label poses and no points.

    The model is in the body-fixed frame and km: each image gets the label attitude and
    ``t = -R c`` with ``c`` the spacecraft position.

    Parameters
    ----------
    ws : Workspace
        Working directory; the image and camera ids come from its database.
    meta : pandas.DataFrame
        Image metadata; database images missing from it are left out.
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    out : pathlib.Path
        Output directory for ``cameras.txt``, ``images.txt`` and an empty
        ``points3D.txt``.

    Returns
    -------
    int
        Number of images written.
    """
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
    """Run ``point_triangulator`` from model ``src`` into ``dst`` (intrinsics fixed)."""
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
    """Run ``bundle_adjuster`` from model ``src`` into ``dst`` (intrinsics fixed; poses
    refined only if ``opt.refine_poses``).
    """
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
    """Map with the label poses as the starting model (``label-poses`` mode).

    Replaces ``sparse/`` with ``sparse/label_poses`` (:func:`write_label_model`), then
    runs ``opt.rounds`` rounds of ``point_triangulator`` (``round<k>_triangulated``)
    followed, if ``opt.refine_poses``, by ``bundle_adjuster`` (``round<k>_adjusted``).
    Only the first round starts from an empty point set.

    Parameters
    ----------
    exe : str
        COLMAP executable.
    ws : Workspace
        Working directory.
    meta : pandas.DataFrame
        Image metadata.
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    opt : ReconstructionOptions
        ``rounds``, ``refine_poses`` and the triangulation options.

    Returns
    -------
    pathlib.Path
        Directory of the final model.
    """
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
    """Run COLMAP's incremental ``mapper`` from scratch (``incremental`` mode).

    Replaces ``sparse/``; each sub-model with at least ``opt.min_model_size`` images is
    written to ``sparse/<k>``.

    Parameters
    ----------
    exe : str
        COLMAP executable.
    ws : Workspace
        Working directory.
    camera : FramingCamera
        Camera model.
    opt : ReconstructionOptions
        Mapper options.
    """
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
    """Convert binary sub-models to TXT under ``model_txt/<name>``.

    Parameters
    ----------
    exe : str
        COLMAP executable.
    ws : Workspace
        Working directory; ``model_txt/`` is replaced.
    subdirs : list of pathlib.Path, optional
        Models to convert; default every sub-directory of ``sparse/``.

    Returns
    -------
    list of dict
        ``name``, ``path`` (TXT directory) and ``num_images``, sorted by ``num_images``,
        largest first.
    """
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
    """Run the COLMAP part of the pipeline: features, matching, mapping and export.

    Parameters
    ----------
    ws : Workspace
        Working directory with the prepared images and masks.
    meta : pandas.DataFrame
        Image metadata.
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    opt : ReconstructionOptions, optional
        Settings; default :class:`ReconstructionOptions`.
    colmap : str, optional
        COLMAP executable, see :func:`colmap_executable`.
    overwrite : bool
        Delete an existing database and start over.
    skip_features : bool
        Reuse the existing database and re-run only the mapping.

    Returns
    -------
    dict
        Also written to ``colmap/reconstruct.json``: ``colmap`` (version), ``options``,
        ``camera`` (``model``, ``params``), ``num_input_images``, ``num_pairs``
        (``None`` with ``skip_features``), ``models`` (:func:`export_models`),
        ``selected_model`` (path of the largest model) and ``timings`` (``features_s``,
        ``matching_s``, ``mapping_s``).

    Raises
    ------
    ValueError
        If ``opt.mode`` is not in :data:`MODES`.
    FileNotFoundError
        If COLMAP or the prepared images are missing.
    FileExistsError
        If the database exists and neither ``overwrite`` nor ``skip_features`` is set.
    RuntimeError
        If a COLMAP call fails or no model is produced.
    """
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
