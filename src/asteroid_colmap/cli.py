"""Command-line interface: ``asteroid-colmap <step> --workdir DIR``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from . import __version__
from .camera import get_camera
from .config import DATASETS, get_dataset
from .workspace import Workspace

log = logging.getLogger("asteroid_colmap")


def cmd_download(args, ws: Workspace):
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
    from .metadata import load_metadata
    from .preprocess import prepare

    ds = get_dataset(args.dataset)
    info = prepare(load_metadata(ws.metadata_csv), ws.images, ws.masks, ds.body,
                   get_camera(ds.camera), flip=args.flip, erode_px=args.erode,
                   rel_threshold=args.threshold, gamma=args.gamma)
    ws.write_json(ws.prepare_json, info)
    log.info("prepared %d images (flip=%s) in %s", len(info["images"]), info["flip"], ws.images)


def cmd_reconstruct(args, ws: Workspace):
    from .metadata import load_metadata
    from .reconstruct import ReconstructionOptions, reconstruct

    ds = get_dataset(args.dataset)
    names = {f.name for f in dataclasses.fields(ReconstructionOptions)}
    opt = ReconstructionOptions(**{k: v for k, v in vars(args).items() if k in names and v is not None})
    reconstruct(ws, load_metadata(ws.metadata_csv), ds.body, get_camera(ds.camera), opt,
                colmap=args.colmap, overwrite=args.overwrite, skip_features=args.skip_features)


def cmd_catalog(args, ws: Workspace):
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
    from .plots import make_all

    paths = make_all(ws, get_dataset(args.dataset), interactive=not args.no_html)
    for p in paths:
        print(p)


def cmd_compare(args, ws: Workspace):
    import json

    from .catalog import compare_catalogs

    other = Workspace(args.other)
    result = compare_catalogs(ws.catalog, other.catalog, min_shared=args.min_shared)
    result["catalogs"] = [str(ws.catalog), str(other.catalog)]
    print(json.dumps(result, indent=2))


def cmd_run(args, ws: Workspace):
    if not args.skip_download:
        cmd_download(args, ws)
    cmd_prepare(args, ws)
    cmd_reconstruct(args, ws)
    cmd_catalog(args, ws)
    cmd_plot(args, ws)


def cmd_info(args, ws: Workspace):
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
    a = s["alignment"]
    print(f"registered images : {s['num_registered_images']}/{s['num_input_images']} "
          f"{s['registered_by_sequence']}")
    print(f"landmarks         : {s['num_landmarks']} ({s['num_curated_landmarks']} curated, "
          f"grades {s['grades']}, {s['num_outliers']} outliers)")
    print(f"observations      : {s['num_observations']} (median track {s['median_track_length']:.0f}, "
          f"reproj {s['median_reproj_error_px']:.2f} px)")
    print(f"alignment         : position RMS {a['position_rms_km']:.3f} km, attitude median "
          f"{a['attitude_median_deg']:.4f} deg, scale {a['scale_km_per_unit']:.4g} km/unit")


def _add_common(p):
    p.add_argument("--workdir", "-w", default="work", help="working directory (default: ./work)")
    p.add_argument("--dataset", "-d", default="vesta-rc3", choices=sorted(DATASETS))


def _add_download(p):
    p.add_argument("--filters", type=int, nargs="+", default=[1], help="FC filter numbers")
    p.add_argument("--subdirs", nargs="+", help="archive sub-directories (default: all)")
    p.add_argument("--max-images", type=int, help="evenly sub-sample to at most N images")
    p.add_argument("--workers", type=int, default=6)


def _add_prepare(p):
    p.add_argument("--flip", default="auto", choices=["auto", "none", "ud", "lr", "rot180"],
                   help="array flip applied to the FITS data (auto: match the label geometry)")
    p.add_argument("--erode", type=float, default=8.0, help="mask erosion from limb/terminator, px")
    p.add_argument("--threshold", type=float, default=0.06, help="body threshold (x p99.5)")
    p.add_argument("--gamma", type=float, default=1.0)


def _add_reconstruct(p):
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
    p.add_argument("--model", help="TXT model directory (default: largest COLMAP model)")
    p.add_argument("--min-track", type=int, default=3)
    p.add_argument("--max-height", type=float, default=60.0,
                   help="flag landmarks further than this from the ellipsoid, km")
    p.add_argument("--spacing", type=float, default=10.0, help="curated-subset cell size, km")


def _add_compare(p):
    p.add_argument("--other", required=True, help="second working directory (same database)")
    p.add_argument("--min-shared", type=int, default=3, help="keypoints two landmarks must share")


def _add_plot(p):
    p.add_argument("--no-html", action="store_true", help="skip the interactive Plotly page")


def build_parser() -> argparse.ArgumentParser:
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
