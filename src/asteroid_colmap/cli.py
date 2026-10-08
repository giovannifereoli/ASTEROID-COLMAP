"""Command-line interface: ``asteroid-colmap <step> --workdir DIR``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .camera import get_camera
from .config import DATASETS, get_dataset
from .workspace import Workspace

log = logging.getLogger("asteroid_colmap")


def cmd_download(args, ws: Workspace):
    """Download the FITS images and labels, then write ``metadata.csv``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``dataset``, ``filters``, ``subdirs``, ``max_images``, ``workers``).
    ws : Workspace
        Working directory; files go to ``ws.raw`` and ``ws.metadata_csv``.
    """
    from .download import download
    from .metadata import build_metadata

    ds = get_dataset(args.dataset)
    download(ds, ws.raw, filters=tuple(args.filters), subdirs=args.subdirs,
             max_images=args.max_images, workers=args.workers)
    df = build_metadata(ws.raw, ds.body, ws.metadata_csv)
    worst = df.rotation_model_check_deg.max()
    log.info("%d images; label vs rotation-model sub-spacecraft check: max %.4f deg", len(df), worst)
    if worst > 0.05:
        log.warning("rotation model disagrees with the labels by %.3f deg", worst)


def cmd_prepare(args, ws: Workspace):
    """Convert the FITS images to 8-bit PNGs and feature masks, and write ``prepare.json``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``dataset``, ``flip``, ``erode``, ``threshold``, ``gamma``).
    ws : Workspace
        Working directory with ``metadata.csv``.
    """
    from .metadata import load_metadata
    from .preprocess import prepare

    ds = get_dataset(args.dataset)
    info = prepare(load_metadata(ws.metadata_csv), ws.images, ws.masks, ds.body,
                   get_camera(ds.camera), flip=args.flip, erode_px=args.erode,
                   rel_threshold=args.threshold, gamma=args.gamma)
    ws.write_json(ws.prepare_json, info)
    log.info("prepared %d images (flip=%s) in %s", len(info["images"]), info["flip"], ws.images)


def cmd_reconstruct(args, ws: Workspace):
    """Run COLMAP feature extraction, matching and mapping.

    Every option whose name matches a :class:`~asteroid_colmap.reconstruct.ReconstructionOptions`
    field, and is not ``None``, overrides that field's default.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``colmap``, ``overwrite``, ``skip_features``, mode and tuning flags).
    ws : Workspace
        Working directory with prepared images.
    """
    from .metadata import load_metadata
    from .reconstruct import ReconstructionOptions, reconstruct

    ds = get_dataset(args.dataset)
    names = {f.name for f in dataclasses.fields(ReconstructionOptions)}
    opt = ReconstructionOptions(**{k: v for k, v in vars(args).items() if k in names and v is not None})
    reconstruct(ws, load_metadata(ws.metadata_csv), ds.body, get_camera(ds.camera), opt,
                colmap=args.colmap, overwrite=args.overwrite, skip_features=args.skip_features)


def cmd_catalog(args, ws: Workspace):
    """Georeference the COLMAP model and write the landmark catalog and ``summary.json``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``model``, ``min_track``, ``max_height``, ``spacing``).
    ws : Workspace
        Working directory with a finished reconstruction.
    """
    from .catalog import build_catalog, summarize, write_catalog
    from .georef import georeference
    from .metadata import load_metadata
    from .model_io import read_model

    ds = get_dataset(args.dataset)
    cam = get_camera(ds.camera)
    recon = ws.read_json(ws.reconstruct_json)
    prep = ws.read_json(ws.prepare_json)
    meta = load_metadata(ws.metadata_csv)
    model = read_model(args.model or recon["selected_model"])
    sim, per_image, align = georeference(model, meta, ds.body, cam)
    cat = build_catalog(model, sim, per_image, meta, ds.body, cam, prep["flip"],
                        min_track=args.min_track, max_abs_height_km=args.max_height,
                        curated_spacing_km=args.spacing)
    write_catalog(ws.catalog, cat)
    summary = summarize(cat, align, sim, recon, prep, curated_spacing_km=args.spacing)
    ws.write_json(ws.catalog / "summary.json", summary)
    _print_summary(summary)


def cmd_plot(args, ws: Workspace):
    """Draw every available figure into ``<workdir>/plots`` and print the paths.

    Selects the non-interactive ``Agg`` backend first, so it runs without a display.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``dataset``, ``no_html``).
    ws : Workspace
        Working directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    from .plots import make_all

    paths = make_all(ws, get_dataset(args.dataset), interactive=not args.no_html)
    for p in paths:
        print(p)


def cmd_compare(args, ws: Workspace):
    """Compare this workspace's catalog with another one and print the result as JSON.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``other``, ``min_shared``).
    ws : Workspace
        First working directory.
    """
    import json

    from .catalog import compare_catalogs

    other = Workspace(args.other)
    result = compare_catalogs(ws.catalog, other.catalog, min_shared=args.min_shared)
    result["catalogs"] = [str(ws.catalog), str(other.catalog)]
    print(json.dumps(result, indent=2))


def cmd_templates(args, ws: Workspace):
    """Build the NCC maplets of the curated landmarks and save ``catalog/templates.npz``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``size``, ``views``, ``maplet_spacing``).
    ws : Workspace
        Working directory with a catalog.
    """
    from .metadata import load_metadata
    from .ncc import build_templates

    ds = get_dataset(args.dataset)
    t = build_templates(ws.catalog, load_metadata(ws.metadata_csv), ds.body, get_camera(ds.camera),
                        size=args.size, views=args.views, spacing_km=args.maplet_spacing)
    t.save(ws.templates)
    log.info("%d maplets (%d x %d samples at %.3f km, median %.0f views) in %s", len(t), t.size,
             t.size, t.spacing_km, float(np.median(t.num_views)), ws.templates)


def _new_images(args, ws: Workspace, run, body):
    """Metadata of the images a ``match`` run measures (see :func:`cmd_match`)."""
    import pandas as pd

    from .download import download, select_evenly
    from .metadata import build_metadata, load_metadata

    if args.catalog_images:
        meta = load_metadata(ws.metadata_csv).drop(columns="time")
    elif args.images:
        meta = pd.concat([build_metadata(Path(p), body) for p in args.images], ignore_index=True)
        meta = meta.drop_duplicates("image").sort_values("utc", ignore_index=True)
    else:
        src = get_dataset(args.source)
        if src.body.name != body.name:
            raise RuntimeError(f"{src.key} images {src.body.name}, the catalog is of {body.name}")
        download(src, run / "raw", filters=tuple(args.filters), subdirs=args.subdirs,
                 max_images=args.max_images, workers=args.workers)
        meta = build_metadata(run / "raw", body)
    keep = select_evenly(list(range(len(meta))), args.max_images)
    return meta.iloc[keep].reset_index(drop=True)


def cmd_match(args, ws: Workspace):
    """Find the catalog landmarks in new images by NCC and correct the image poses.

    The images are the label/FITS pairs under ``--images`` (any download folders), the
    catalog's own images with ``--catalog-images`` (a self-consistency check: each image is
    matched without its own reference views), or else they are downloaded from ``--source``
    into ``navigation/<name>/raw``. Writes ``metadata.csv``, ``matches.csv``, ``poses.csv``,
    ``summary.json`` and navigation figures 01-04 to ``navigation/<name>/``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (image source, ``name`` and :class:`~asteroid_colmap.ncc.MatchOptions`
        overrides).
    ws : Workspace
        Working directory with ``catalog/templates.npz``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from .ncc import MatchOptions, Templates, match_images, summarize_matches
    from .plots import make_navigation_plots

    ds = get_dataset(args.dataset)
    cam = get_camera(ds.camera)
    if not ws.templates.exists():
        raise FileNotFoundError(f"{ws.templates} not found - run 'asteroid-colmap templates' first")
    t = Templates.load(ws.templates)
    name = args.name or ("catalog" if args.catalog_images else
                         "local" if args.images else args.source)
    run = ws.navigation / name
    run.mkdir(parents=True, exist_ok=True)
    meta = _new_images(args, ws, run, ds.body)
    meta.to_csv(run / "metadata.csv", index=False)
    names = {f.name for f in dataclasses.fields(MatchOptions)}
    opt = MatchOptions(**{k: v for k, v in vars(args).items() if k in names and v is not None})
    cams = pd.read_csv(ws.catalog / "cameras.csv")
    matches, poses = match_images(meta, t, ds.body, cam, opt, catalog_cameras=cams)
    if not len(poses):
        raise RuntimeError("no image could be matched")
    matches.to_csv(run / "matches.csv", index=False)
    poses.to_csv(run / "poses.csv", index=False)
    summary = {"name": name, "images": int(len(meta)), "matching": summarize_matches(matches, poses, t, opt)}
    ws.write_json(run / "summary.json", summary)
    m = summary["matching"]
    log.info("%d/%d images with a pose; median %d inliers, RMS %.2f -> %.2f px", m["num_images_with_pose"],
             m["num_images"], m["median_inliers_per_image"], m["median_prefit_rms_px"],
             m["median_postfit_rms_px"])
    for p in make_navigation_plots(run / "plots", meta, matches, poses, t, cam, target=ds.body.name):
        print(p)


def _run_dir(ws: Workspace, name: str | None) -> Path:
    """Folder of a ``match`` run; without ``name`` the only run there is."""
    if name:
        run = ws.navigation / name
    else:
        runs = sorted(p for p in ws.navigation.glob("*") if (p / "matches.csv").exists())
        if len(runs) != 1:
            raise FileNotFoundError(f"pass --name, one of: {[p.name for p in runs]}" if runs else
                                    f"no matching run in {ws.navigation} - run 'asteroid-colmap match'")
        run = runs[0]
    if not (run / "matches.csv").exists():
        raise FileNotFoundError(f"{run / 'matches.csv'} not found - run 'asteroid-colmap match' first")
    return run


def cmd_pose(args, ws: Workspace):
    """Estimate each image's pose relative to the body from its NCC landmark matches.

    Solves position and attitude without the label pose (:mod:`~asteroid_colmap.pose`), fits a
    smooth arc per sequence, compares both with the label and, for catalog images, with the
    COLMAP pose. Writes ``relative_pose.csv``, ``trajectory.csv``, a ``relative_pose`` section
    of ``summary.json`` and navigation figure 05 to the run folder.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (``name``, ``degree`` and :class:`~asteroid_colmap.pose.PoseOptions`
        overrides).
    ws : Workspace
        Working directory with a ``match`` run.
    """
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from .metadata import load_metadata
    from .plots import plot_relative_pose
    from .pose import PoseOptions, fit_trajectory, relative_poses, summarize_relative

    ds = get_dataset(args.dataset)
    run = _run_dir(ws, args.name)
    names = {f.name for f in dataclasses.fields(PoseOptions)}
    opt = PoseOptions(**{k: v for k, v in vars(args).items() if k in names and v is not None})
    cams = ws.catalog / "cameras.csv"
    rel = relative_poses(pd.read_csv(run / "matches.csv"), load_metadata(run / "metadata.csv"),
                         ds.body, get_camera(ds.camera), opt,
                         ncc_poses=pd.read_csv(run / "poses.csv"),
                         catalog_cameras=pd.read_csv(cams) if cams.exists() else None)
    traj = fit_trajectory(rel, ds.body, degree=args.degree)
    rel.to_csv(run / "relative_pose.csv", index=False)
    traj.to_csv(run / "trajectory.csv", index=False)
    summary = ws.read_json(run / "summary.json") if (run / "summary.json").exists() else {}
    summary["relative_pose"] = s = summarize_relative(rel, traj, opt)
    ws.write_json(run / "summary.json", summary)
    log.info("%d/%d images solved; median RMS %.2f px, formal 1-sigma %.2f km along the range",
             s["num_images_with_pose"], s["num_images"], s["median_rms_px"],
             s.get("median_sigma_range_km", float("nan")))
    if s["num_images_with_pose"]:
        print(plot_relative_pose(run / "plots", rel, traj if len(traj) else None, ds.body.name))


def cmd_run(args, ws: Workspace):
    """Run download (unless ``--skip-download``), prepare, reconstruct, catalog and plot.

    Parameters
    ----------
    args : argparse.Namespace
        Union of the options of every step.
    ws : Workspace
        Working directory.
    """
    if not args.skip_download:
        cmd_download(args, ws)
    cmd_prepare(args, ws)
    cmd_reconstruct(args, ws)
    cmd_catalog(args, ws)
    cmd_plot(args, ws)


def cmd_info(args, ws: Workspace):
    """Print the known datasets and, if present, the workspace's catalog summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options (unused apart from the workspace).
    ws : Workspace
        Working directory.
    """
    for key, ds in DATASETS.items():
        cam = get_camera(ds.camera)
        print(f"{key}: {ds.description}")
        print(f"  archive : {ds.base_url}")
        print(f"  subdirs : {', '.join(ds.subdirs)}")
        print(f"  body    : {ds.body.name}, radii {ds.body.radii_km} km, frame {ds.body.frame_name}")
        print(f"  camera  : {cam.name}; COLMAP {cam.colmap_model} {cam.colmap_params_str()}")
    if (ws.catalog / "summary.json").exists():
        _print_summary(ws.read_json(ws.catalog / "summary.json"))


def _print_summary(s: dict):
    """Print the key numbers of a catalog ``summary.json`` dictionary."""
    a = s["alignment"]
    print(f"registered images : {s['num_registered_images']}/{s['num_input_images']} "
          f"{s['registered_by_sequence']}")
    print(f"landmarks         : {s['num_landmarks']} ({s['num_curated_landmarks']} curated, "
          f"grades {s['grades']}, {s['num_outliers']} outliers)")
    print(f"observations      : {s['num_observations']} (median track {s['median_track_length']:.0f}, "
          f"reproj {s['median_reproj_error_px']:.2f} px)")
    print(f"alignment         : position RMS {a['position_rms_km']:.3f} km, attitude median "
          f"{a['attitude_median_deg']:.4f} deg, scale {a['scale_km_per_unit']:.4g} km/unit")
    if "position_range_rms_km" in a:
        print(f"                    position RMS {a['position_range_rms_km']:.3f} km along the line "
              f"of sight, {a['position_lateral_rms_km']:.3f} km sideways; boresight median "
              f"{a['boresight_median_px']:.1f} px")
    if "label_pointing_offset_by_sequence_px" in s:
        by_seq = ", ".join(f"{k} ({u:+.1f}, {v:+.1f})"
                           for k, (u, v) in s["label_pointing_offset_by_sequence_px"].items())
        print(f"label poses       : landmarks {s['label_pointing_offset_median_px']:.2f} px (median) "
              f"from where they are seen; median (u, v) px {by_seq}")


def _add_common(p):
    """Add ``--workdir`` and ``--dataset`` to a sub-command parser."""
    p.add_argument("--workdir", "-w", default="work", help="working directory (default: ./work)")
    p.add_argument("--dataset", "-d", default="vesta-rc3", choices=sorted(DATASETS))


def _add_download(p):
    """Add the options of the ``download`` step."""
    p.add_argument("--filters", type=int, nargs="+", default=[1], help="FC filter numbers")
    p.add_argument("--subdirs", nargs="+", help="archive sub-directories (default: all)")
    p.add_argument("--max-images", type=int, help="evenly sub-sample to at most N images")
    p.add_argument("--workers", type=int, default=6)


def _add_prepare(p):
    """Add the options of the ``prepare`` step."""
    p.add_argument("--flip", default="auto", choices=["auto", "none", "ud", "lr", "rot180"],
                   help="array flip applied to the FITS data (auto: match the label geometry)")
    p.add_argument("--erode", type=float, default=8.0, help="mask erosion from limb/terminator, px")
    p.add_argument("--threshold", type=float, default=0.06, help="body threshold (x p99.5)")
    p.add_argument("--gamma", type=float, default=1.0)


def _add_reconstruct(p):
    """Add the options of the ``reconstruct`` step (defaults live in ``ReconstructionOptions``)."""
    p.add_argument("--colmap", help="path to the colmap executable")
    p.add_argument("--overwrite", action="store_true", help="delete an existing database")
    p.add_argument("--skip-features", action="store_true", help="reuse database, re-run mapper")
    p.add_argument("--mode", choices=["label-poses", "incremental"],
                   help="label-poses (default): seed COLMAP with the PDS label poses; "
                        "incremental: COLMAP mapper from scratch")
    p.add_argument("--no-refine-poses", dest="refine_poses", action="store_false", default=None,
                   help="label-poses: keep the label poses fixed (triangulate only)")
    p.add_argument("--rounds", type=int, help="label-poses: triangulate/adjust iterations (2)")
    p.add_argument("--matching", choices=["pairs", "exhaustive", "sequential"])
    p.add_argument("--max-view-angle", dest="max_view_angle_deg", type=float)
    p.add_argument("--max-features", dest="max_num_features", type=int)
    p.add_argument("--peak-threshold", type=float)
    p.add_argument("--init-min-tri-angle", dest="init_min_tri_angle_deg", type=float)
    p.add_argument("--max-reproj-error", dest="filter_max_reproj_error_px", type=float)
    p.add_argument("--num-threads", type=int)
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)


def _add_catalog(p):
    """Add the options of the ``catalog`` step."""
    p.add_argument("--model", help="TXT model directory (default: largest COLMAP model)")
    p.add_argument("--min-track", type=int, default=3)
    p.add_argument("--max-height", type=float, default=60.0,
                   help="flag landmarks further than this from the ellipsoid, km")
    p.add_argument("--spacing", type=float, default=10.0, help="curated-subset cell size, km")


def _add_compare(p):
    """Add the options of the ``compare`` step."""
    p.add_argument("--other", required=True, help="second working directory (same database)")
    p.add_argument("--min-shared", type=int, default=3, help="keypoints two landmarks must share")


def _add_templates(p):
    """Add the options of the ``templates`` step."""
    p.add_argument("--size", type=int, default=31, help="maplet width in samples (odd)")
    p.add_argument("--views", type=int, default=4, help="reference views per landmark")
    p.add_argument("--maplet-spacing", type=float,
                   help="maplet sample spacing, km (default: median landmark GSD)")


def _add_match(p):
    """Add the options of the ``match`` step (matching defaults live in ``MatchOptions``)."""
    p.add_argument("--name", help="run folder under <workdir>/navigation "
                                  "(default: the source dataset, 'local' or 'catalog')")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--source", default="vesta-opnav", choices=sorted(DATASETS),
                     help="dataset to download the new images from")
    src.add_argument("--images", nargs="+", metavar="DIR",
                     help="folders with downloaded FIT + LBL pairs (no download)")
    src.add_argument("--catalog-images", action="store_true",
                     help="the catalog's own images, each without its own reference views")
    _add_download(p)
    p.add_argument("--estimate-position", action="store_true", default=None,
                   help="also fit a position correction (weak from far away)")
    p.add_argument("--min-ncc", type=float, help="peak correlation to accept a match (0.8)")
    p.add_argument("--search", dest="search_px", type=int, help="search half-width, px (6)")
    p.add_argument("--coarse-search", dest="coarse_search_px", type=int,
                   help="pass-1 search half-width, px (20)")


def _add_pose(p):
    """Add the options of the ``pose`` step (defaults live in ``PoseOptions``)."""
    p.add_argument("--name", help="run folder under <workdir>/navigation (default: the only one)")
    p.add_argument("--degree", type=int, default=2, help="polynomial degree of the smoothed arc")
    p.add_argument("--sigma", dest="sigma_px", type=float, help="measurement noise, px (0.3)")
    p.add_argument("--ransac-threshold", dest="ransac_threshold_px", type=float,
                   help="RANSAC inlier threshold, px (2)")


def _add_plot(p):
    """Add the options of the ``plot`` step."""
    p.add_argument("--no-html", action="store_true", help="skip the interactive Plotly page")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``asteroid-colmap`` argument parser with one sub-command per step.

    Returns
    -------
    argparse.ArgumentParser
        Parser whose sub-commands set ``args.func`` to the matching ``cmd_*`` function.
    """
    parser = argparse.ArgumentParser(
        prog="asteroid-colmap",
        description="Dawn FC images of Vesta -> COLMAP -> body-fixed landmark catalog + plots.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    steps = [
        ("download", "download FITS + labels and build metadata.csv", cmd_download, [_add_download]),
        ("prepare", "FITS -> 8-bit PNG + masks, orientation check", cmd_prepare, [_add_prepare]),
        ("reconstruct", "COLMAP features, matching and mapping", cmd_reconstruct, [_add_reconstruct]),
        ("catalog", "georeference and write the landmark catalog", cmd_catalog, [_add_catalog]),
        ("plot", "figures in <workdir>/plots", cmd_plot, [_add_plot]),
        ("run", "all of the above", cmd_run,
         [_add_download, _add_prepare, _add_reconstruct, _add_catalog, _add_plot]),
        ("compare", "compare two catalogs landmark by landmark (e.g. both mapping modes)",
         cmd_compare, [_add_compare]),
        ("templates", "NCC maplets of the curated landmarks", cmd_templates, [_add_templates]),
        ("match", "find the landmarks in new images by NCC and correct their poses", cmd_match,
         [_add_match]),
        ("pose", "relative position and attitude from the matches, smoothed arc", cmd_pose,
         [_add_pose]),
        ("info", "datasets, camera model and the latest summary", cmd_info, []),
    ]
    for name, help_, func, adders in steps:
        p = sub.add_parser(name, help=help_)
        _add_common(p)
        for add in adders:
            add(p)
        if name == "run":
            p.add_argument("--skip-download", action="store_true")
        p.set_defaults(func=func)
    return parser


def main(argv=None) -> int:
    """Entry point of the ``asteroid-colmap`` command.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 when a step fails with a missing file, an existing
        file or a runtime error (the error is logged instead of raised).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        args.func(args, Workspace(args.workdir))
    except (FileNotFoundError, FileExistsError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
