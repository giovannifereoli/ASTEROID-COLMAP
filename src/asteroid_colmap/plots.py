"""Figures for every pipeline stage (written to ``<workdir>/plots``)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (FixedLocator, NullLocator, ScalarFormatter,  # noqa: E402
                               StrMethodFormatter)
from PIL import Image  # noqa: E402

from .camera import get_camera  # noqa: E402
from .config import Dataset  # noqa: E402
from .metadata import load_metadata  # noqa: E402
from .workspace import Workspace  # noqa: E402

log = logging.getLogger(__name__)

# palette (validated: scripts/validate_palette.js, light mode, all pairs for the first three)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#1c5cab", "#3987e5", "#9ec5f4", "#f0efec",
                     "#f2a3a2", "#e34948", "#b83232", "#7a1f1f"])
DPI = 150


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.5, "font.family": "sans-serif",
        "figure.titlesize": 13, "figure.titleweight": "bold",
    })


def _seq_colors(sequences) -> dict[str, str]:
    return {s: CATEGORICAL[i % len(CATEGORICAL)] for i, s in enumerate(sequences)}


def _legend_handles(colors: dict[str, str], marker="o"):
    return [Line2D([], [], color=c, marker=marker, linestyle="-" if marker is None else "",
                   markersize=7, linewidth=1.5, label=s) for s, c in colors.items()]


def _time_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b %Y"))


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def _suptitle(fig, title: str, subtitle: str | None = None):
    fig.text(0.01, 1.0, title, ha="left", va="bottom", fontsize=13, weight="bold", color=INK,
             transform=fig.transFigure)
    if subtitle:
        fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=9.5, color=INK2,
                 transform=fig.transFigure)


# ---------------------------------------------------------------- inputs & geometry

def plot_montage(ws: Workspace, meta: pd.DataFrame, colors, n: int = 24, ncols: int = 8) -> Path:
    idx = np.unique(np.linspace(0, len(meta) - 1, min(n, len(meta))).round().astype(int))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.7, nrows * 2.15 + 0.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        r = meta.iloc[i]
        ax.imshow(Image.open(ws.images / r["image"]), cmap="gray", vmin=0, vmax=255)
        ax.add_patch(plt.Rectangle((0, 0.965), 1, 0.035, transform=ax.transAxes, color=colors[r["sequence"]]))
        ax.set_title(f"{r['sequence']}  {r['time']:%H:%M}\nsub-SC {r['subsc_lat_deg']:+.0f}°, "
                     f"{r['subsc_lon_model_deg']:.0f}°E", fontsize=7.5, color=INK2, weight="normal",
                     loc="center")
    _suptitle(fig, "Input images", f"{len(idx)} of {len(meta)} prepared frames (8-bit stretch, "
              "orientation-corrected); colour bar = sequence")
    fig.legend(handles=_legend_handles(colors, marker="s"), loc="lower right",
               bbox_to_anchor=(0.99, 0.985), ncol=len(colors))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=1 - 0.75 / fig.get_figheight(),
                        wspace=0.04, hspace=0.32)
    return _save(fig, ws.plots / "01_montage.png")


def plot_geometry(ws: Workspace, meta: pd.DataFrame, colors) -> Path:
    panels = [("subsc_lon_model_deg", "Sub-spacecraft longitude (°E)"),
              ("subsc_lat_deg", "Sub-spacecraft latitude (°)"),
              ("range_km", "Range to Vesta centre (km)"),
              ("phase_deg", "Phase angle (°)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
    for ax, (col, label) in zip(axes.flat, panels):
        for seq, grp in meta.groupby("sequence", sort=False):
            ax.plot(grp["time"], grp[col], "o", color=colors[seq], markersize=4)
        ax.set_title(label, fontsize=10)
        _time_axis(ax)
    axes[0, 0].set_ylim(0, 360)
    axes[0, 0].set_yticks(range(0, 361, 90))
    for ax in axes[1]:
        ax.set_xlabel("UTC")
    _suptitle(fig, "Observation geometry", "Two full rotations of Vesta seen from ~5,500 km; "
              "RC3B views the southern hemisphere at low phase")
    fig.legend(handles=_legend_handles(colors), loc="upper right", bbox_to_anchor=(0.99, 1.04),
               ncol=len(colors))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, ws.plots / "02_geometry.png")


def plot_orientation(ws: Workspace, ds: Dataset, meta: pd.DataFrame, prep: dict) -> Path | None:
    from .preprocess import FLIPS, apply_flip, body_mask, load_fits, predicted_lit_mask

    scores = pd.DataFrame(prep.get("orientation_scores") or [])
    cam = get_camera(ds.camera)
    picks = [0, len(meta) // 2, len(meta) - 1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for ax, i in zip(axes[:3], picks):
        r = meta.iloc[i]
        img = apply_flip(load_fits(r["fit_path"]), prep["flip"])
        ax.imshow(np.array(Image.open(ws.images / r["image"])), cmap="gray")
        pred = predicted_lit_mask(r, ds.body, cam, step=2)
        ext = (0, cam.width, cam.height, 0)
        ax.contour(body_mask(img, prep["rel_threshold"]).astype(float), levels=[0.5],
                   colors=[CATEGORICAL[1]], linewidths=1.5)
        ax.contour(pred.astype(float), levels=[0.5], colors=[CATEGORICAL[0]], linewidths=1.5,
                   linestyles="--", extent=ext, origin="upper")
        ax.set_title(f"{r['sequence']} {r['time']:%H:%M}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(handles=[Line2D([], [], color=CATEGORICAL[1], label="observed body mask"),
                            Line2D([], [], color=CATEGORICAL[0], ls="--",
                                   label="label geometry + ellipsoid")],
                   loc="lower left", fontsize=8, facecolor=SURFACE, framealpha=0.9, frameon=True)
    ax = axes[3]
    if len(scores):
        mean = scores.groupby("flip")["iou"].mean().reindex(FLIPS)
        ax.barh(mean.index, mean.values, color=[CATEGORICAL[0] if f == prep["flip"] else AXIS
                                                for f in mean.index], height=0.6)
        for y, v in enumerate(mean.values):
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()
        ax.set_xlabel("mean silhouette IoU")
        ax.set_title("Array flip vs label geometry", fontsize=10)
        ax.grid(axis="y", visible=False)
    else:
        ax.axis("off")
    _suptitle(fig, "Orientation check", f"Selected flip: {prep['flip']!r} - the FITS rows are "
              "stored bottom-up relative to the camera frame")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, ws.plots / "03_orientation_check.png")


# ---------------------------------------------------------------- reconstruction

def plot_features(ws: Workspace, cams: pd.DataFrame, obs: pd.DataFrame, lm: pd.DataFrame,
                  colors) -> Path:
    track = lm.set_index("landmark_id").track_length
    picks = (cams[cams.registered].sort_values("num_landmarks", ascending=False)
             .drop_duplicates("sequence").sort_values("utc"))
    fig, axes = plt.subplots(1, len(picks), figsize=(5.4 * len(picks) + 1.0, 6.0), squeeze=False)
    vmax = float(np.percentile(track, 99))
    for ax, (_, r) in zip(axes[0], picks.iterrows()):
        o = obs[obs.image == r["image"]]
        ax.imshow(np.array(Image.open(ws.images / r["image"])), cmap="gray")
        sc = ax.scatter(o.u - 0.5, o.v - 0.5, c=o.landmark_id.map(track), cmap=SEQUENTIAL,
                        norm=LogNorm(3, vmax), s=4, linewidths=0)
        ax.set_title(f"{r['sequence']} · {r['image'][:-4]}\n{len(o):,} landmarks", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.subplots_adjust(left=0.01, right=0.9, bottom=0.01, top=0.86, wspace=0.03)
    cax = fig.add_axes((0.915, 0.12, 0.014, 0.64))
    cb = fig.colorbar(sc, cax=cax)
    ticks = [t for t in (3, 5, 10, 20, 30, 50, 100) if t <= vmax]
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(ScalarFormatter())
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.set_label("track length (images)")
    cb.outline.set_edgecolor(AXIS)
    _suptitle(fig, "Triangulated keypoints", "Landmark observations on the best-connected image "
              "of each sequence")
    return _save(fig, ws.plots / "04_features.png")


def plot_covisibility(ws: Workspace, cams: pd.DataFrame, obs: pd.DataFrame) -> Path:
    order = cams.sort_values("utc").image.tolist()
    pos = {n: i for i, n in enumerate(order)}
    o = obs[["landmark_id", "image"]].assign(i=obs.image.map(pos))
    inc = pd.crosstab(o.landmark_id, o.i).reindex(columns=range(len(order)), fill_value=0)
    A = (inc.T.values @ inc.values).astype(float)
    np.fill_diagonal(A, np.nan)
    A[A == 0] = np.nan
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(A, cmap=SEQUENTIAL, norm=LogNorm(1, np.nanmax(A)), interpolation="nearest")
    seq = cams.sort_values("utc").sequence.to_numpy()
    edges = np.flatnonzero(seq[1:] != seq[:-1]) + 0.5
    for e in edges:
        ax.axhline(e, color=INK2, lw=0.8)
        ax.axvline(e, color=INK2, lw=0.8)
    bounds = np.r_[-0.5, edges, len(seq) - 0.5]
    centres = (bounds[:-1] + bounds[1:]) / 2
    labels = [seq[int(np.ceil(b + 0.5))] for b in bounds[:-1]]
    ax.set_xticks(centres, labels)
    ax.set_yticks(centres, labels)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.yaxis.set_major_locator(FixedLocator([t for t in (1, 10, 100, 1000, 10000) if t <= np.nanmax(A)]))
    cb.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.set_label("shared landmarks")
    cb.outline.set_edgecolor(AXIS)
    ax.set_xlabel("image (time order)")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.9)
    _suptitle(fig, "Co-visibility", "Landmarks shared by each image pair; off-diagonal blocks "
              "tie the two sequences together")
    return _save(fig, ws.plots / "05_covisibility.png")


def plot_alignment(ws: Workspace, cams: pd.DataFrame, colors, summary: dict) -> Path:
    c = cams[cams.registered].copy()
    c["time"] = pd.to_datetime(c.utc)
    c["boresight_px"] = np.hypot(c.boresight_du_px, c.boresight_dv_px)
    panels = [("position_residual_km", "Camera position residual after similarity fit (km)"),
              ("boresight_px", "COLMAP vs label boresight offset (px)"),
              ("twist_deg", "COLMAP vs label twist about the boresight (°)")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True)
    for ax, (col, label) in zip(axes, panels):
        for seq, grp in c.groupby("sequence", sort=False):
            ax.plot(grp.time, grp[col], "o-", color=colors[seq], markersize=3.5, lw=1.2)
        ax.set_title(label, fontsize=10)
    axes[2].axhline(0, color=AXIS, lw=1)
    _time_axis(axes[2])
    axes[2].set_xlabel("UTC")
    a = summary["alignment"]
    _suptitle(fig, "Georeferencing residuals",
              f"position RMS {a['position_rms_km']:.2f} km over {a['num_used_in_fit']} images; "
              f"median attitude difference {a['attitude_median_deg'] * 3600:.0f}\" "
              f"({a['boresight_median_px']:.1f} px)")
    fig.legend(handles=_legend_handles(colors), loc="upper right", bbox_to_anchor=(0.99, 1.03),
               ncol=len(colors))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, ws.plots / "06_alignment.png")


# ---------------------------------------------------------------- catalog

def _height_norm(h: pd.Series) -> TwoSlopeNorm:
    lim = float(np.nanpercentile(np.abs(h), 98))
    return TwoSlopeNorm(0.0, -lim, lim)


def plot_pointcloud(ws: Workspace, lm: pd.DataFrame, max_points: int = 60000) -> Path:
    g = lm[~lm.outlier]
    if len(g) > max_points:
        g = g.sample(max_points, random_state=0)
    norm = _height_norm(g.height_km)
    views = [(15, 300, "seen from 300°E, 15°N"), (15, 120, "seen from 120°E, 15°N"),
             (-60, 0, "seen from below (south)")]
    fig = plt.figure(figsize=(15, 4.9))
    lim = np.abs(g[["x_km", "y_km", "z_km"]].to_numpy()).max()
    for k, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d", computed_zorder=False)
        sc = ax.scatter(g.x_km, g.y_km, g.z_km, c=g.height_km, cmap=DIVERGING, norm=norm, s=0.6,
                        linewidths=0, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-lim, lim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=10, loc="center", y=1.0)
    fig.subplots_adjust(left=0.0, right=0.9, bottom=0.0, top=0.88, wspace=0.0)
    cax = fig.add_axes((0.91, 0.12, 0.012, 0.66))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("height above reference ellipsoid (km)")
    cb.outline.set_edgecolor(AXIS)
    _suptitle(fig, "Landmark point cloud (body-fixed, Claudia double-prime)",
              f"{len(g):,} landmarks coloured by height relative to the "
              "286.3 × 278.6 × 223.2 km ellipsoid")
    return _save(fig, ws.plots / "07_pointcloud.png")


def plot_map(ws: Workspace, lm: pd.DataFrame, curated: pd.DataFrame) -> Path:
    g = lm[~lm.outlier]
    fig, grid = plt.subplots(2, 2, figsize=(11.5, 10.2), width_ratios=(1, 0.025))
    axes = grid[:, 0]
    grid[1, 1].axis("off")
    ax = axes[0]
    order = np.argsort(np.abs(g.height_km.to_numpy()))
    sc = ax.scatter(g.lon_deg.to_numpy()[order], g.lat_deg.to_numpy()[order],
                    c=g.height_km.to_numpy()[order], cmap=DIVERGING, norm=_height_norm(g.height_km),
                    s=1.2, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, cax=grid[0, 1])
    cb.set_label("height above ellipsoid (km)")
    cb.outline.set_edgecolor(AXIS)
    ax.set_title(f"All landmarks ({len(g):,}), coloured by height", fontsize=10)
    ax = axes[1]
    grade_colors = {"A": "#0d366b", "B": "#3987e5", "C": "#9ec5f4"}
    c = curated[~curated.outlier] if "outlier" in curated else curated
    for grade in ("C", "B", "A"):  # best grade drawn last, on top
        sel = c[c.grade == grade]
        if sel.empty:
            continue
        ax.scatter(sel.lon_deg, sel.lat_deg, s=7, color=grade_colors[grade], linewidths=0.3,
                   edgecolors=SURFACE, label=f"{grade}  ({len(sel):,})", rasterized=True)
    handles, labels = ax.get_legend_handles_labels()
    grid[1, 1].legend(handles[::-1], labels[::-1], title="grade", loc="upper left",
                      bbox_to_anchor=(0.0, 1.0), markerscale=2.2, borderaxespad=0, alignment="left")
    ax.set_title(f"Curated subset ({len(c):,}): best grade A/B landmark per "
                 f"{_curated_spacing(ws):g} km equal-area cell", fontsize=10)
    for ax in axes:
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xticks(range(0, 361, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.set_aspect("equal")
        ax.set_xlabel("east longitude (°)")
        ax.set_ylabel("planetocentric latitude (°)")
    _suptitle(fig, "Landmark map", "Equirectangular, Claudia double-prime frame; the north is "
              "unlit during approach (polar night)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, ws.plots / "08_landmark_map.png")


def _curated_spacing(ws: Workspace) -> float:
    try:
        return float(ws.read_json(ws.catalog / "summary.json").get("curated_spacing_km") or 10.0)
    except (FileNotFoundError, ValueError):
        return 10.0


def plot_quality(ws: Workspace, lm: pd.DataFrame) -> Path:
    from .catalog import GRADES

    g = lm[~lm.outlier]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    specs = [
        ("track_length", "Track length (images)", np.arange(2.5, min(g.track_length.max(), 80) + 1.5, 1),
         [GRADES["B"][0], GRADES["A"][0]]),
        ("reproj_error_px", "Mean reprojection error (px)", np.linspace(0, g.reproj_error_px.max(), 50),
         [GRADES["A"][2], GRADES["B"][2]]),
        ("max_tri_angle_deg", "Max triangulation angle (°)",
         np.linspace(0, g.max_tri_angle_deg.max(), 50), [GRADES["B"][1], GRADES["A"][1]]),
        ("height_km", "Height above ellipsoid (km)", np.linspace(*np.nanpercentile(g.height_km, [0.5, 99.5]), 60), []),
    ]
    for ax, (col, title, bins, thresholds) in zip(axes.flat, specs):
        ax.hist(g[col], bins=bins, color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=0.5)
        for t in thresholds:
            ax.axvline(t, color=INK2, lw=1, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("landmarks")
    counts = lm.grade.value_counts().reindex(["A", "B", "C"], fill_value=0)
    _suptitle(fig, "Landmark quality",
              f"grades A/B/C: {counts['A']:,} / {counts['B']:,} / {counts['C']:,}; dashed lines = grade "
              "thresholds (A: track ≥ 8, angle ≥ 15°, error ≤ 1 px; B: ≥ 4, ≥ 5°, ≤ 2 px)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, ws.plots / "09_quality.png")


def plot_radius(ws: Workspace, lm: pd.DataFrame) -> Path:
    g = lm[~lm.outlier]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    hb = ax.hexbin(g.lat_deg, g.height_km, gridsize=(90, 40), cmap=SEQUENTIAL, mincnt=1, bins="log",
                   linewidths=0)
    bins = np.arange(-90, 91, 5)
    mid = (bins[:-1] + bins[1:]) / 2
    med = g.groupby(pd.cut(g.lat_deg, bins), observed=False).height_km.median().to_numpy()
    ax.plot(mid, med, color=INK, lw=1.5, label="median per 5° band")
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.legend(loc="upper right")
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("landmarks per cell")
    cb.outline.set_edgecolor(AXIS)
    ax.set_xlabel("planetocentric latitude (°)")
    ax.set_ylabel("height above ellipsoid (km)")
    _suptitle(fig, "Radius residuals vs latitude", "Height of each landmark above the reference "
              "ellipsoid; the low southern latitudes lie in the Rheasilvia basin")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, ws.plots / "10_radius_vs_latitude.png")


def plot_chips(ws: Workspace, curated: pd.DataFrame, obs: pd.DataFrame, cams: pd.DataFrame,
               n_landmarks: int = 8, n_views: int = 7, half: int = 24) -> Path:
    cand = curated[curated.grade == "A"]
    if len(cand) < n_landmarks:
        cand = curated
    # spread the picks over longitude
    cand = cand.assign(lon_bin=(cand.lon_deg // (360 / n_landmarks)).astype(int))
    picks = (cand.sort_values("track_length", ascending=False).drop_duplicates("lon_bin")
             .sort_values("lon_deg").head(n_landmarks))
    if len(picks) < n_landmarks:
        extra = cand[~cand.landmark_id.isin(picks.landmark_id)].nlargest(n_landmarks - len(picks),
                                                                          "track_length")
        picks = pd.concat([picks, extra]).sort_values("lon_deg")
    times = cams.set_index("image").utc
    size = Image.open(ws.images / obs.image.iloc[0]).size  # (width, height)
    inside = ((obs.u - 0.5 >= half) & (obs.u - 0.5 <= size[0] - 1 - half)
              & (obs.v - 0.5 >= half) & (obs.v - 0.5 <= size[1] - 1 - half))
    obs = obs[inside]
    n_inside = obs.groupby("landmark_id").size()
    cand = cand[cand.landmark_id.map(n_inside).fillna(0) >= min(n_views, 4)]
    picks = picks[picks.landmark_id.isin(cand.landmark_id)]
    if len(picks) < n_landmarks:
        extra = cand[~cand.landmark_id.isin(picks.landmark_id)].nlargest(n_landmarks - len(picks),
                                                                          "track_length")
        picks = pd.concat([picks, extra]).sort_values("lon_deg")
    fig, axes = plt.subplots(len(picks), n_views, figsize=(n_views * 1.35, len(picks) * 1.5))
    axes = np.atleast_2d(axes)
    cache: dict[str, np.ndarray] = {}
    for row, (_, lmk) in zip(axes, picks.iterrows()):
        o = obs[obs.landmark_id == lmk.landmark_id].assign(t=lambda d: d.image.map(times)).sort_values("t")
        sel = o.iloc[np.unique(np.linspace(0, len(o) - 1, min(n_views, len(o))).round().astype(int))]
        for ax in row:
            ax.axis("off")
        for ax, (_, ob) in zip(row, sel.iterrows()):
            img = cache.setdefault(ob.image, np.array(Image.open(ws.images / ob.image)))
            c, r = int(round(ob.u - 0.5)), int(round(ob.v - 0.5))
            chip = img[r - half:r + half + 1, c - half:c + half + 1]
            ax.imshow(chip, cmap="gray", vmin=np.percentile(chip, 1), vmax=np.percentile(chip, 99.5))
            ax.plot(half, half, "+", color=CATEGORICAL[1], markersize=9, mew=1.2)
            ax.set_title(pd.Timestamp(ob.t).strftime("%d %H:%M"), fontsize=7, color=INK2, loc="center")
        row[0].text(-0.12, 0.5, f"{lmk.landmark_id}\n{lmk.lat_deg:+.1f}°, {lmk.lon_deg:.1f}°E\n"
                    f"{lmk.track_length} views", transform=row[0].transAxes, ha="right", va="center",
                    fontsize=7.5, color=INK)
    _suptitle(fig, "Landmark chips", f"{2 * half + 1}×{2 * half + 1} px around each observation, "
              "earliest to latest view (+ = measured keypoint); one row per grade A landmark")
    fig.tight_layout(rect=(0.08, 0, 1, 0.95))
    return _save(fig, ws.plots / "11_landmark_chips.png")


def plot_interactive(ws: Workspace, lm: pd.DataFrame, cams: pd.DataFrame, max_points: int = 80000):
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.info("plotly not installed - skipping the interactive page (pip install plotly)")
        return None
    g = lm[~lm.outlier]
    if len(g) > max_points:
        g = g.sample(max_points, random_state=0)
    lim = float(np.nanpercentile(np.abs(g.height_km), 98))
    colorscale = [[i / 8, c] for i, c in enumerate(
        ["#0d366b", "#1c5cab", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#b83232", "#7a1f1f"])]
    hover = (g.landmark_id + "<br>lat " + g.lat_deg.round(2).astype(str) + "°, lon "
             + g.lon_deg.round(2).astype(str) + "°E<br>h " + g.height_km.round(2).astype(str)
             + " km<br>track " + g.track_length.astype(str) + ", grade " + g.grade)
    fig = go.Figure(go.Scatter3d(
        x=g.x_km, y=g.y_km, z=g.z_km, mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=1.6, color=g.height_km, colorscale=colorscale, cmin=-lim, cmax=lim,
                    colorbar=dict(title="height (km)"))))
    fig.update_layout(
        title="Vesta landmark catalog (body-fixed km)", paper_bgcolor=SURFACE,
        scene=dict(aspectmode="data", xaxis_title="x (km)", yaxis_title="y (km)", zaxis_title="z (km)"),
        margin=dict(l=0, r=0, t=40, b=0), font=dict(color=INK))
    path = ws.plots / "pointcloud.html"
    fig.write_html(path, include_plotlyjs="cdn")
    log.info("wrote %s", path)
    return path


def make_all(ws: Workspace, ds: Dataset, interactive: bool = True) -> list[Path]:
    _style()
    ws.plots.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(ws.metadata_csv)
    colors = _seq_colors(dict.fromkeys(meta.sequence))
    out = [plot_geometry(ws, meta, colors)]
    if ws.prepare_json.exists():
        prep = ws.read_json(ws.prepare_json)
        out += [plot_montage(ws, meta, colors), plot_orientation(ws, ds, meta, prep)]
    if (ws.catalog / "landmarks.csv").exists():
        lm = pd.read_csv(ws.catalog / "landmarks.csv")
        curated = pd.read_csv(ws.catalog / "landmarks_curated.csv")
        obs = pd.read_csv(ws.catalog / "observations.csv")
        cams = pd.read_csv(ws.catalog / "cameras.csv")
        summary = ws.read_json(ws.catalog / "summary.json")
        out += [
            plot_features(ws, cams, obs, lm, colors),
            plot_covisibility(ws, cams[cams.registered], obs),
            plot_alignment(ws, cams, colors, summary),
            plot_pointcloud(ws, lm),
            plot_map(ws, lm, curated),
            plot_quality(ws, lm),
            plot_radius(ws, lm),
            plot_chips(ws, curated, obs, cams),
        ]
        if interactive:
            out.append(plot_interactive(ws, lm, cams))
    return [p for p in out if p is not None]
