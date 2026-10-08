"""Figures for every pipeline stage (written to ``<workdir>/plots``).

Every public ``plot_*`` function draws one figure, saves it as a PNG in
:attr:`Workspace.plots` and returns the path. The house style is applied with
:func:`matplotlib.pyplot.rc_context` inside each call, so importing this module or calling a
plot function never changes global matplotlib settings, which keeps notebooks clean. The
backend is not forced either: the command-line interface selects ``Agg`` itself.

Figures
-------
01 montage, 02 geometry, 03 orientation check, 04 features, 05 co-visibility,
06 alignment, 07 globe views, 08 landmark maps, 09 quality, 10 radius vs latitude,
11 landmark chips, plus ``pointcloud.html`` (interactive, needs plotly).

Navigation figures
------------------
Written by :func:`make_navigation_plots` for an NCC matching run on new images
(``<workdir>/navigation/<name>/plots``): 01 matches in one image, 02 maplet / template /
image chips, 03 pointing error and fit residuals, 04 correction, residuals and counts per
image, 05 relative pose (:mod:`~asteroid_colmap.pose`). These functions take the run's tables
and an output directory, not a workspace.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter
from PIL import Image

from .camera import get_camera
from .catalog import GRADES
from .config import Dataset
from .geometry import ellipsoid_radius, latlon_to_unit
from .metadata import load_metadata
from .workspace import Workspace

log = logging.getLogger(__name__)

# palette (validated: scripts/validate_palette.js, light mode, all pairs for the first three)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL_STOPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_STOPS = ["#0d366b", "#1c5cab", "#3987e5", "#9ec5f4", "#f0efec",
                   "#f2a3a2", "#e34948", "#b83232", "#7a1f1f"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_STOPS)
# the same ramp without its lightest step, for small marks drawn on the light surface
SEQUENTIAL_MARKS = LinearSegmentedColormap.from_list("seq_blue_marks", SEQUENTIAL_STOPS[1:])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING_STOPS)
# grades are ordered (A best), so they take steps of the sequential ramp, darkest = best
GRADE_COLORS = {"A": "#0d366b", "B": "#3987e5", "C": "#9ec5f4"}
DPI = 150

STYLE = {
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
}
"""matplotlib rcParams of the house style, applied by :func:`_styled`."""


def _styled(func):
    """Run a plotting function inside ``plt.rc_context(STYLE)``.

    Parameters
    ----------
    func : callable
        Function that creates, saves and closes its figures.

    Returns
    -------
    callable
        Wrapped function with the same signature and docstring.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Call ``func`` with the house style active."""
        with plt.rc_context(STYLE):
            return func(*args, **kwargs)
    return wrapper


def _seq_colors(sequences) -> dict[str, str]:
    """Assign a categorical colour to each sequence, in fixed order.

    Parameters
    ----------
    sequences : iterable of str
        Sequence names in display order (for example ``RC3``, ``RC3B``).

    Returns
    -------
    dict
        ``{sequence: hex colour}``. The first three sequences take the CATEGORICAL hues in
        order; any further sequence is drawn in the muted grey instead of a cycled hue.
    """
    names = list(sequences)
    return {s: CATEGORICAL[i] if i < len(CATEGORICAL) else MUTED for i, s in enumerate(names)}


def _legend_handles(colors: dict[str, str], marker="o", labels: dict[str, str] | None = None):
    """Build legend proxies for a ``{name: colour}`` mapping.

    Parameters
    ----------
    colors : dict
        ``{name: colour}``, in legend order.
    marker : str or None
        Marker of each proxy; ``None`` draws a line instead.
    labels : dict, optional
        ``{name: label text}``; defaults to the names themselves.

    Returns
    -------
    list of matplotlib.lines.Line2D
    """
    labels = labels or {}
    return [Line2D([], [], color=c, marker=marker, linestyle="-" if marker is None else "",
                   markersize=7, linewidth=1.5, label=labels.get(s, s)) for s, c in colors.items()]


def _time_col(df: pd.DataFrame) -> str:
    """Name of the timestamp column: ``time`` (metadata) or ``utc`` (cameras.csv)."""
    return "time" if "time" in df else "utc"


def _hours_since_start(df: pd.DataFrame, time_col: str | None = None) -> pd.Series:
    """Elapsed time of each row since the first image of its sequence.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a ``sequence`` column and a timestamp column.
    time_col : str, optional
        Timestamp column; defaults to :func:`_time_col`.

    Returns
    -------
    pandas.Series
        Hours since the sequence start, aligned with ``df``.
    """
    t = pd.to_datetime(df[time_col or _time_col(df)])
    return (t - t.groupby(df["sequence"]).transform("min")).dt.total_seconds() / 3600.0


def _seq_labels(df: pd.DataFrame, time_col: str | None = None) -> dict[str, str]:
    """Legend labels that name each sequence and its start time.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a ``sequence`` column and a timestamp column.
    time_col : str, optional
        Timestamp column; defaults to :func:`_time_col`.

    Returns
    -------
    dict
        ``{sequence: "RC3 (from 24 Jul 2011 06:00 UTC)"}``.
    """
    t = pd.to_datetime(df[time_col or _time_col(df)])
    t0 = t.groupby(df["sequence"], sort=False).min()
    return {s: f"{s}  (from {v:%d %b %Y %H:%M} UTC)" for s, v in t0.items()}


def _break_wraps(x, y, jump: float = 180.0):
    """Insert NaN gaps where a longitude series wraps around 0/360°.

    Parameters
    ----------
    x, y : array_like
        Longitude (degrees) and the matching ordinate.
    jump : float
        Smallest step in ``x`` that counts as a wrap.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)`` with NaN inserted before every wrap, so a line plot does not draw
        a segment across the whole map.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    cut = np.flatnonzero(np.abs(np.diff(x)) > jump) + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)


def _polar_night(subsolar_lat_deg) -> tuple[float, float] | None:
    """Latitude band that never sees the Sun during a full rotation.

    Parameters
    ----------
    subsolar_lat_deg : array_like
        Sub-solar latitudes of the images (degrees).

    Returns
    -------
    tuple of float or None
        ``(low, high)`` latitude limits in degrees, or ``None`` when the Sun is within 1°
        of the equator or no latitude is available.
    """
    s = np.asarray(subsolar_lat_deg, float)
    s = s[np.isfinite(s)]
    if s.size == 0 or abs(s.mean()) < 1.0:
        return None
    m = float(s.mean())
    return (90.0 + m, 90.0) if m < 0 else (-90.0, -90.0 + m)


def _save(fig, path: Path) -> Path:
    """Save a figure at :data:`DPI`, close it and return the path."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def _suptitle(fig, title: str, subtitle: str | None = None):
    """Write a left-aligned bold title and an optional subtitle at the top of a figure.

    The title sits just above the figure area (``bbox_inches="tight"`` keeps it), and the
    subtitle hangs just below the top edge, so callers leave about 0.4 inch free there.
    """
    fig.text(0.01, 1.0, title, ha="left", va="bottom", fontsize=13, weight="bold", color=INK,
             transform=fig.transFigure)
    if subtitle:
        fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=9.5, color=INK2,
                 transform=fig.transFigure)


def _log_ticks(cb, vmin: float, vmax: float):
    """Put plain-number ticks (1, 2, 5, 10, ...) on a logarithmic colorbar.

    Parameters
    ----------
    cb : matplotlib.colorbar.Colorbar
        Colorbar whose long axis is logarithmic.
    vmin, vmax : float
        Data range of the colorbar; the tick density adapts to the number of decades.
    """
    decades = np.log10(max(vmax, 1e-9) / max(vmin, 1e-9))
    mantissas = (1, 2, 3, 5) if decades <= 1.5 else (1, 2, 5) if decades <= 2.2 else \
        (1, 3) if decades <= 3 else (1,)
    ticks = [m * 10.0 ** e for e in range(-2, 8) for m in mantissas]
    ticks = [t for t in ticks if vmin * 0.999 <= t <= vmax * 1.001]
    axis = cb.ax.yaxis if cb.orientation == "vertical" else cb.ax.xaxis
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(StrMethodFormatter("{x:,g}"))
    axis.set_minor_locator(NullLocator())


def _colorbar(fig, mappable, label: str, **kw):
    """Add a colorbar with the house outline colour and a label; ``kw`` go to ``fig.colorbar``."""
    cb = fig.colorbar(mappable, **kw)
    cb.set_label(label)
    cb.outline.set_edgecolor(AXIS)
    return cb


# ---------------------------------------------------------------- inputs & geometry

@_styled
def plot_montage(ws: Workspace, meta: pd.DataFrame, colors, n: int = 24, ncols: int = 8) -> Path:
    """Figure 01: a grid of prepared input images, evenly spaced in time.

    Parameters
    ----------
    ws : Workspace
        Workspace with prepared PNGs in ``ws.images``.
    meta : pandas.DataFrame
        Image metadata from :func:`~asteroid_colmap.metadata.load_metadata`.
    colors : dict
        ``{sequence: colour}`` from :func:`_seq_colors`.
    n : int
        Maximum number of images shown.
    ncols : int
        Images per row.

    Returns
    -------
    pathlib.Path
        ``plots/01_montage.png``.
    """
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


@_styled
def plot_geometry(ws: Workspace, meta: pd.DataFrame, colors) -> Path:
    """Figure 02: sub-spacecraft ground track, range and phase angle.

    The left panel maps the sub-spacecraft point of every image on a latitude/longitude
    grid and shades the polar-night band; the right panels show range and phase against
    hours since the start of each sequence, so the sequences overlay.

    Parameters
    ----------
    ws : Workspace
        Output workspace.
    meta : pandas.DataFrame
        Image metadata (:func:`~asteroid_colmap.metadata.load_metadata`) or the catalog's
        ``cameras.csv``. Needs ``sequence``, ``time`` or ``utc``, ``subsc_lat_deg``,
        ``subsc_lon_model_deg`` or ``subsc_lon_deg``, ``range_km`` and ``phase_deg``; the
        polar-night band is drawn only when ``subsolar_lat_deg`` is present.
    colors : dict
        ``{sequence: colour}`` from :func:`_seq_colors`.

    Returns
    -------
    pathlib.Path
        ``plots/02_geometry.png``.
    """
    tcol = _time_col(meta)
    lon_col = "subsc_lon_model_deg" if "subsc_lon_model_deg" in meta else "subsc_lon_deg"
    m = meta.assign(hours=_hours_since_start(meta, tcol), lon=meta[lon_col] % 360)
    m = m.sort_values(["sequence", tcol])
    fig = plt.figure(figsize=(14, 5.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93), w_pad=0.08, wspace=0.06)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.85, 1))
    ax_t = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1], sharex=ax_r)

    night = _polar_night(meta["subsolar_lat_deg"]) if "subsolar_lat_deg" in meta else None
    if night:
        ax_t.axhspan(*night, color=GRID, lw=0, zorder=0)
        hemi = "north" if night[1] == 90 else "south"
        edge = night[0] if hemi == "north" else night[1]
        ax_t.text(4, np.mean(night), f"polar night ({hemi} of {abs(edge):.0f}°"
                  f"{'N' if hemi == 'north' else 'S'})", va="center", fontsize=8.5, color=INK2)
    for seq, grp in m.groupby("sequence", sort=False):
        x, y = _break_wraps(grp["lon"], grp["subsc_lat_deg"])
        ax_t.plot(x, y, "-", color=colors[seq], lw=1.0)
        ax_t.plot(grp["lon"], grp["subsc_lat_deg"], "o", color=colors[seq], markersize=3.5)
        for ax, col in ((ax_r, "range_km"), (ax_p, "phase_deg")):
            ax.plot(grp["hours"], grp[col], "o-", color=colors[seq], markersize=3.5, lw=1.0)
    ax_t.set_xlim(0, 360)
    ax_t.set_ylim(-90, 90)
    ax_t.set_xticks(range(0, 361, 30))
    ax_t.set_yticks(range(-90, 91, 30))
    ax_t.set_aspect("equal")
    ax_t.set_xlabel("east longitude (°)")
    ax_t.set_ylabel("planetocentric latitude (°)")
    ax_t.set_title("Sub-spacecraft point", fontsize=10)
    ax_r.set_title("Range to Vesta centre (km)", fontsize=10)
    ax_p.set_title("Phase angle (°)", fontsize=10)
    ax_r.tick_params(labelbottom=False)
    ax_p.set_xlabel("hours since sequence start")

    parts = []
    for seq, grp in m.groupby("sequence", sort=False):
        lat, ph = grp["subsc_lat_deg"], grp["phase_deg"]
        parts.append(f"{seq} {lat.min():+.0f}° to {lat.max():+.0f}° latitude at "
                     f"{ph.min():.0f}–{ph.max():.0f}° phase over {grp['hours'].max():.1f} h")
    _suptitle(fig, "Observation geometry",
              f"{len(m)} images from {m.range_km.min():,.0f}–{m.range_km.max():,.0f} km; "
              + "; ".join(parts) + ". Sub-spacecraft longitude falls as Vesta rotates.")
    fig.legend(handles=_legend_handles(colors, labels=_seq_labels(m, tcol)), loc="lower right",
               bbox_to_anchor=(0.99, 1.0), ncol=len(colors))
    return _save(fig, ws.plots / "02_geometry.png")


@_styled
def plot_orientation(ws: Workspace, ds: Dataset, meta: pd.DataFrame, prep: dict) -> Path | None:
    """Figure 03: observed body silhouettes against the label-predicted lit ellipsoid.

    Three images (first, middle, last) show the observed body mask and the silhouette
    predicted from the label geometry; a bar chart ranks the candidate array flips by
    mean silhouette IoU.

    Parameters
    ----------
    ws : Workspace
        Workspace with prepared PNGs.
    ds : Dataset
        Dataset definition (camera and body).
    meta : pandas.DataFrame
        Image metadata with ``fit_path``.
    prep : dict
        Contents of ``prepare.json`` (``flip``, ``rel_threshold``, ``orientation_scores``).

    Returns
    -------
    pathlib.Path
        ``plots/03_orientation_check.png``.
    """
    from .preprocess import FLIPS, apply_flip, body_mask, load_fits, predicted_lit_mask

    scores = pd.DataFrame(prep.get("orientation_scores") or [])
    cam = get_camera(ds.camera)
    picks = [0, len(meta) // 2, len(meta) - 1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
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
    fig.legend(handles=[Line2D([], [], color=CATEGORICAL[1], label="observed body mask"),
                        Line2D([], [], color=CATEGORICAL[0], ls="--",
                               label="label geometry + ellipsoid")],
               loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2)
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
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return _save(fig, ws.plots / "03_orientation_check.png")


# ---------------------------------------------------------------- reconstruction

@_styled
def plot_features(ws: Workspace, cams: pd.DataFrame, obs: pd.DataFrame, lm: pd.DataFrame,
                  colors) -> Path:
    """Figure 04: landmark observations on the best-connected image of each sequence.

    Parameters
    ----------
    ws : Workspace
        Workspace with prepared PNGs.
    cams : pandas.DataFrame
        Catalog ``cameras.csv``.
    obs : pandas.DataFrame
        Catalog ``observations.csv`` (``u``, ``v`` in COLMAP pixel-edge convention).
    lm : pandas.DataFrame
        Catalog ``landmarks.csv``; the dots are coloured by track length.
    colors : dict
        ``{sequence: colour}``; kept for a uniform call signature.

    Returns
    -------
    pathlib.Path
        ``plots/04_features.png``.
    """
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
    cb = _colorbar(fig, sc, "track length (images)", cax=fig.add_axes((0.915, 0.12, 0.014, 0.64)))
    _log_ticks(cb, 3, vmax)
    _suptitle(fig, "Triangulated keypoints", "Landmark observations on the best-connected image "
              "of each sequence")
    return _save(fig, ws.plots / "04_features.png")


@_styled
def plot_covisibility(ws: Workspace, cams: pd.DataFrame, obs: pd.DataFrame) -> Path:
    """Figure 05: number of landmarks shared by every pair of images.

    Parameters
    ----------
    ws : Workspace
        Output workspace.
    cams : pandas.DataFrame
        Registered rows of ``cameras.csv``; sets the time order and sequence blocks.
    obs : pandas.DataFrame
        Catalog ``observations.csv``.

    Returns
    -------
    pathlib.Path
        ``plots/05_covisibility.png``.
    """
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
    cb = _colorbar(fig, im, "shared landmarks", ax=ax, shrink=0.8, pad=0.02)
    _log_ticks(cb, 1, np.nanmax(A))
    ax.set_xlabel("image (time order)")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.9)
    _suptitle(fig, "Co-visibility", "Landmarks shared by each image pair; off-diagonal blocks "
              "tie the two sequences together")
    return _save(fig, ws.plots / "05_covisibility.png")


@_styled
def plot_alignment(ws: Workspace, cams: pd.DataFrame, colors, summary: dict) -> Path:
    """Figure 06: per-image residuals between the COLMAP and label camera poses.

    Four panels against hours since sequence start: the camera-position residual after the
    similarity fit, split into its line-of-sight (signed) and sideways parts; the pointing
    difference in pixels, both at the boresight and at the catalog landmarks (median offset
    of the landmarks projected with the label pose from where they are measured); and the
    twist about the boresight. A sideways shift paired with a turn that keeps the target
    centred moves the boresight but hardly the landmarks, so the two pixel curves differ.

    Parameters
    ----------
    ws : Workspace
        Output workspace.
    cams : pandas.DataFrame
        Catalog ``cameras.csv``.
    colors : dict
        ``{sequence: colour}`` from :func:`_seq_colors`.
    summary : dict
        Catalog ``summary.json``; its ``alignment`` block feeds the subtitle.

    Returns
    -------
    pathlib.Path
        ``plots/06_alignment.png``.
    """
    c = cams[cams.registered].copy()
    c["hours"] = _hours_since_start(c, "utc")
    c["boresight_px"] = np.hypot(c.boresight_du_px, c.boresight_dv_px)
    c["landmark_px"] = np.hypot(c.label_offset_u_px, c.label_offset_v_px)
    c["twist_arcsec"] = c.twist_deg * 3600.0
    fit = c.used_in_fit.astype(bool)
    rms = {k: float(np.sqrt(np.mean(c.loc[fit, k] ** 2)))
           for k in ("position_residual_km", "position_range_km", "position_lateral_km")}
    open_marker = {"linestyle": "--", "markerfacecolor": SURFACE}
    panels = [
        ([("position_range_km", {})], "km", "Camera position residual along the line of sight "
         f"(RMS {rms['position_range_km']:.2f} km; + = COLMAP farther than the label)"),
        ([("position_lateral_km", {})], "km",
         f"Camera position residual sideways (RMS {rms['position_lateral_km']:.2f} km)"),
        ([("boresight_px", {}), ("landmark_px", open_marker)], "px",
         f"Pointing difference: boresight, COLMAP vs label (median {c.boresight_px.median():.1f} px); "
         f"landmarks under the label pose (median {c.landmark_px.median():.1f} px)"),
        ([("twist_arcsec", {})], "arcsec", "Twist about the boresight, COLMAP vs label "
         f"(median |twist| {c.twist_arcsec.abs().median():.0f} arcsec)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.955))
    for ax, (series, unit, title) in zip(axes, panels):
        for col, kw in series:
            for seq, grp in c.sort_values("utc").groupby("sequence", sort=False):
                ax.plot(grp.hours, grp[col], color=colors[seq], markersize=3.5, lw=1.0,
                        **{"marker": "o", "linestyle": "-", **kw})
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
    for ax in (axes[1], axes[2]):
        ax.set_ylim(bottom=0)
    for ax in (axes[0], axes[3]):
        ax.axhline(0, color=AXIS, lw=1)
    axes[2].legend(handles=[Line2D([], [], color=INK2, marker="o", markersize=5, lw=1.0,
                                   label="boresight"),
                            Line2D([], [], color=INK2, marker="o", markersize=5, lw=1.0,
                                   label="landmarks", **open_marker)],
                   loc="upper right", fontsize=8.5, frameon=False)
    axes[3].set_xlabel("hours since sequence start")
    a = summary["alignment"]
    n_fit = "" if fit.all() else f" over {int(fit.sum())} of {len(c)} images"
    _suptitle(fig, "Georeferencing residuals",
              f"position RMS {rms['position_residual_km']:.2f} km{n_fit} "
              f"({rms['position_range_km']:.2f} km line of sight, "
              f"{rms['position_lateral_km']:.2f} km sideways); attitude "
              f"{a['attitude_median_deg'] * 3600:.0f}\" = {c.boresight_px.median():.1f} px at the "
              f"boresight, {c.landmark_px.median():.1f} px at the landmarks")
    fig.legend(handles=_legend_handles(colors, labels=_seq_labels(c, "utc")), loc="lower right",
               bbox_to_anchor=(0.99, 1.0), ncol=len(colors))
    return _save(fig, ws.plots / "06_alignment.png")


# ---------------------------------------------------------------- catalog

def _height_norm(h: pd.Series) -> TwoSlopeNorm:
    """Symmetric diverging norm centred on 0 km, clipped at the 98th percentile of ``|h|``."""
    lim = float(np.nanpercentile(np.abs(h), 98))
    return TwoSlopeNorm(0.0, -lim, lim)


def _view_basis(lat0: float, lon0: float):
    """Orthographic camera basis for a view centred on ``(lat0, lon0)``.

    Parameters
    ----------
    lat0, lon0 : float
        Planetocentric latitude and east longitude of the view centre (degrees).

    Returns
    -------
    tuple of numpy.ndarray
        ``(d, right, up)``: unit vector towards the viewer, then the screen axes. The
        basis is right-handed (``right × up = d``), so the view is not mirrored; ``up``
        points to the north pole, or to 0°E when looking straight at a pole.
    """
    d = latlon_to_unit(lat0, lon0)
    lo = np.radians(lon0)
    right = np.array([-np.sin(lo), np.cos(lo), 0.0])
    up = np.cross(d, right)
    return d, right, up / np.linalg.norm(up)


def _globe_view(ax, xyz: np.ndarray, values, cmap, norm, radii, lat0: float, lon0: float,
                s: float = 0.8):
    """Draw landmarks on an orthographic globe with graticule, limb and labels.

    Points on the far side (outward ellipsoid normal facing away from the viewer) are
    culled, and the rest are drawn far to near so nearer points cover farther ones.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target 2-D axes; its frame is switched off.
    xyz : numpy.ndarray
        ``(N, 3)`` body-fixed landmark positions (km).
    values : array_like
        ``(N,)`` values mapped through ``cmap`` and ``norm``.
    cmap, norm : matplotlib colormap and norm
    radii : sequence of float
        Reference ellipsoid semi-axes ``(a, b, c)`` in km.
    lat0, lon0 : float
        View centre (degrees).
    s : float
        Marker area (points²).

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter, for a colorbar.
    """
    a = np.asarray(radii, float)
    d, right, up = _view_basis(lat0, lon0)
    facing = (xyz / a ** 2) @ d > 0
    p, v = xyz[facing], np.asarray(values)[facing]
    order = np.argsort(p @ d)
    sc = ax.scatter((p @ right)[order], (p @ up)[order], c=v[order], cmap=cmap, norm=norm, s=s,
                    linewidths=0, rasterized=True, zorder=2)

    def surface(lat, lon):
        """Project ellipsoid surface points onto the view plane; far-side points become NaN."""
        X = ellipsoid_radius(lat, lon, a)[..., None] * latlon_to_unit(lat, lon)
        vis = (X / a ** 2) @ d > 0
        x, y = X @ right, X @ up
        x[~vis] = np.nan
        y[~vis] = np.nan
        return x, y, vis

    halo = [pe.withStroke(linewidth=2.2, foreground=SURFACE)]
    for lat in range(-60, 61, 30):
        x, y, _ = surface(np.full(361, lat), np.arange(361.0))
        ax.plot(x, y, color=INK, lw=0.7 if lat == 0 else 0.4, alpha=0.35, zorder=3)
    for lon in range(0, 360, 30):
        x, y, _ = surface(np.linspace(-90, 90, 181), np.full(181, lon))
        ax.plot(x, y, color=INK, lw=0.7 if lon == 0 else 0.4, alpha=0.35, zorder=3)
    for lon in range(0, 360, 90):
        x, y, vis = surface(np.array([0.0]), np.array([float(lon)]))
        X = ellipsoid_radius(0.0, lon, a) * latlon_to_unit(0.0, lon)
        if vis[0] and (X / np.linalg.norm(X)) @ d > 0.25:
            ax.text(x[0], y[0], f"{lon}°E", fontsize=7.5, color=INK2, ha="center", va="bottom",
                    path_effects=halo, zorder=4)
    for lat, name in ((-90.0, "S"), (90.0, "N")):
        x, y, vis = surface(np.array([lat]), np.array([0.0]))
        if vis[0] and abs(np.sin(np.radians(lat)) * d[2]) > 0.25:
            ax.plot(x, y, "+", color=INK, markersize=6, mew=1.0, zorder=4)
            ax.text(x[0], y[0], f" {name}", fontsize=8, color=INK, ha="left", va="center",
                    path_effects=halo, zorder=4)

    w = d / a
    w /= np.linalg.norm(w)
    ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p1 = np.cross(w, ref)
    p1 /= np.linalg.norm(p1)
    q1 = np.cross(w, p1)
    t = np.linspace(0, 2 * np.pi, 721)[:, None]
    limb = a * (np.cos(t) * p1 + np.sin(t) * q1)
    lx, ly = limb @ right, limb @ up
    ax.plot(lx, ly, color=AXIS, lw=0.8, zorder=3)
    xl, yl = 1.06 * np.abs(lx).max(), 1.06 * np.abs(ly).max()
    ax.set_xlim(-xl, xl)
    ax.set_ylim(-yl, yl)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def _lat_label(lat: float) -> str:
    """Format a latitude as ``30°S`` / ``0°`` / ``15°N``."""
    return "0°" if lat == 0 else f"{abs(lat):g}°{'N' if lat > 0 else 'S'}"


@_styled
def plot_pointcloud(ws: Workspace, lm: pd.DataFrame, radii=(286.3, 278.6, 223.2),
                    max_points: int = 60000, lat0: float = -30.0) -> Path:
    """Figure 07: four orthographic globe views of the landmarks, coloured by height.

    Parameters
    ----------
    ws : Workspace
        Output workspace.
    lm : pandas.DataFrame
        Landmarks (``landmarks.csv`` or the curated subset) with ``x_km``, ``y_km``,
        ``z_km``, ``height_km`` and ``outlier``.
    radii : sequence of float
        Reference ellipsoid semi-axes in km (used for culling, graticule and limb).
    max_points : int
        Random subsample size, for speed.
    lat0 : float
        Latitude of the four view centres (degrees); the centres are 0, 90, 180 and 270°E.

    Returns
    -------
    pathlib.Path
        ``plots/07_pointcloud.png``.
    """
    g = lm[~lm.outlier]
    if len(g) > max_points:
        g = g.sample(max_points, random_state=0)
    norm = _height_norm(g.height_km)
    xyz = g[["x_km", "y_km", "z_km"]].to_numpy()
    H = 3.8
    top = 1 - 0.78 / H
    fig, axes = plt.subplots(1, 4, figsize=(15.5, H))
    fig.subplots_adjust(left=0.005, right=0.915, bottom=0.02, top=top, wspace=0.02)
    for ax, lon0 in zip(axes, (0, 90, 180, 270)):
        sc = _globe_view(ax, xyz, g.height_km.to_numpy(), DIVERGING, norm, radii, lat0, lon0)
        ax.set_anchor("S")
        # one shared baseline for the titles, whatever each globe's aspect ratio
        box = ax.get_position(original=True)
        fig.text((box.x0 + box.x1) / 2, top + 0.01, f"centred on {_lat_label(lat0)}, {lon0}°E",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    _colorbar(fig, sc, "height above ellipsoid (km)", cax=fig.add_axes((0.93, 0.08, 0.01, 0.6)),
              extend="both")
    a, b, c = radii
    _suptitle(fig, "Landmark globe (body-fixed, Claudia double-prime)",
              f"{len(g):,} landmarks coloured by height above the {a:g} × {b:g} × {c:g} km "
              "ellipsoid; orthographic views, graticule every 30° (equator and prime meridian "
              "thicker). The blank north was in polar night.")
    return _save(fig, ws.plots / "07_pointcloud.png")


def _polar_xy(lat, lon):
    """South-polar azimuthal-equidistant coordinates: radius 90° + latitude, 0°E up.

    East longitude runs clockwise, as seen from below the south pole.
    """
    r = 90.0 + np.asarray(lat, float)
    lo = np.radians(lon)
    return r * np.sin(lo), r * np.cos(lo)


def _polar_frame(ax):
    """Draw latitude rings, longitude spokes and labels on a south-polar panel."""
    t = np.radians(np.arange(361))
    az = np.radians(15.0)
    for lat in (-80, -60, -40, -20, 0):
        r = 90 + lat
        ax.plot(r * np.sin(t), r * np.cos(t), color=INK, lw=0.8 if lat == 0 else 0.4, alpha=0.35,
                zorder=3)
        if lat:  # the outer ring is the equator; the spoke labels sit just outside it
            ax.text(r * np.sin(az), r * np.cos(az), f"{abs(lat)}°S", fontsize=7.5, color=INK2,
                    ha="center", va="center", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=SURFACE)])
    for lon in range(0, 360, 30):
        lo = np.radians(lon)
        ax.plot([0, 90 * np.sin(lo)], [0, 90 * np.cos(lo)], color=INK,
                lw=0.7 if lon == 0 else 0.4, alpha=0.35, zorder=3)
        ax.text(97 * np.sin(lo), 97 * np.cos(lo), f"{lon}°E", fontsize=7.5, color=INK2,
                ha="center", va="center")
    ax.set_xlim(-104, 104)
    ax.set_ylim(-104, 104)
    ax.set_aspect("equal")
    ax.axis("off")


def _map_axes(ax, title: str, night: tuple[float, float] | None):
    """Format an equirectangular longitude/latitude panel and shade polar night."""
    if night:
        ax.axhspan(*night, color=GRID, lw=0, zorder=0)
        ax.text(4, np.mean(night), "polar night", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(0, 361, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_aspect("equal")
    ax.set_xlabel("east longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.set_title(title, fontsize=10)


@_styled
def plot_map(ws: Workspace, lm: pd.DataFrame, curated: pd.DataFrame,
             spacing_km: float | None = None, night_lat_deg: tuple[float, float] | None = None
             ) -> Path:
    """Figure 08: landmark maps.

    Top left: every landmark on an equirectangular map, coloured by height. Bottom left:
    the curated subset coloured by track length. Right: the southern hemisphere in a
    south-polar azimuthal-equidistant view, coloured by height (the Rheasilvia basin).

    Parameters
    ----------
    ws : Workspace
        Output workspace; ``catalog/summary.json`` supplies the curated cell size when
        ``spacing_km`` is not given.
    lm : pandas.DataFrame
        All landmarks (needs ``lat_deg``, ``lon_deg``, ``height_km``, ``outlier``).
    curated : pandas.DataFrame
        Curated landmarks (needs ``track_length``).
    spacing_km : float, optional
        Curated cell size, shown in the panel title.
    night_lat_deg : tuple of float, optional
        Latitude band to shade as polar night (see :func:`_polar_night`).

    Returns
    -------
    pathlib.Path
        ``plots/08_landmark_map.png``.
    """
    spacing_km = spacing_km if spacing_km is not None else _curated_spacing(ws)
    g = lm[~lm.outlier]
    c = curated[~curated.outlier] if "outlier" in curated else curated
    norm = _height_norm(g.height_km)
    fig = plt.figure(figsize=(15.5, 9.0), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94), wspace=0.04)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.15, 1))
    ax_h, ax_t, ax_p = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    order = np.argsort(np.abs(g.height_km.to_numpy()))  # extreme heights drawn on top
    gh = g.iloc[order]
    sc = ax_h.scatter(gh.lon_deg, gh.lat_deg, c=gh.height_km, cmap=DIVERGING, norm=norm, s=1.0,
                      linewidths=0, rasterized=True)
    _map_axes(ax_h, f"All landmarks ({len(g):,}), coloured by height", night_lat_deg)
    _colorbar(fig, sc, "height above ellipsoid (km)", ax=ax_h, pad=0.01, fraction=0.03,
              extend="both")

    tmax = float(max(c.track_length.max(), 4))
    ct = c.sort_values("track_length")
    sc = ax_t.scatter(ct.lon_deg, ct.lat_deg, c=ct.track_length, cmap=SEQUENTIAL_MARKS,
                      norm=LogNorm(max(float(c.track_length.min()), 1.0), tmax), s=5, linewidths=0,
                      rasterized=True)
    _map_axes(ax_t, f"Curated subset ({len(c):,}): best grade A/B landmark per "
              f"{spacing_km:g} km equal-area cell", night_lat_deg)
    cb = _colorbar(fig, sc, "track length (images)", ax=ax_t, pad=0.01, fraction=0.03)
    _log_ticks(cb, float(c.track_length.min()), tmax)

    s = gh[gh.lat_deg <= 0]
    x, y = _polar_xy(s.lat_deg, s.lon_deg)
    sc = ax_p.scatter(x, y, c=s.height_km, cmap=DIVERGING, norm=norm, s=1.2, linewidths=0,
                      rasterized=True, zorder=2)
    _polar_frame(ax_p)
    ax_p.set_title(f"Southern hemisphere from below ({len(s):,} landmarks)", fontsize=10)
    _colorbar(fig, sc, "height above ellipsoid (km)", ax=ax_p, orientation="horizontal",
              shrink=0.7, pad=0.01, aspect=35, extend="both")
    _suptitle(fig, "Landmark maps", "Claudia double-prime frame, planetocentric latitude. Left: "
              "equirectangular; right: south-polar azimuthal equidistant (rings every 20°, outer "
              "ring = equator, longitude clockwise). The Rheasilvia basin fills the south-polar "
              "view.")
    return _save(fig, ws.plots / "08_landmark_map.png")


def _curated_spacing(ws: Workspace) -> float:
    """Curated cell size (km) from ``catalog/summary.json``, or 10 km when unavailable."""
    try:
        return float(ws.read_json(ws.catalog / "summary.json").get("curated_spacing_km") or 10.0)
    except (FileNotFoundError, ValueError):
        return 10.0


@_styled
def plot_quality(ws: Workspace, lm: pd.DataFrame) -> Path:
    """Figure 09: distributions of the landmark quality metrics, stacked by grade.

    Panels: track length (log count axis), mean reprojection error, maximum
    triangulation angle and height. Dashed lines mark the grade thresholds of
    :data:`asteroid_colmap.catalog.GRADES`. Each x axis is clipped at a high percentile
    so a few extreme values do not squash the histogram.

    Parameters
    ----------
    ws : Workspace
        Output workspace.
    lm : pandas.DataFrame
        Landmarks with ``track_length``, ``reproj_error_px``, ``max_tri_angle_deg``,
        ``height_km``, ``grade`` and ``outlier``.

    Returns
    -------
    pathlib.Path
        ``plots/09_quality.png``.
    """
    g = lm[~lm.outlier]
    grades = [k for k in ("A", "B", "C") if (g.grade == k).any()]
    t_hi = max(float(np.percentile(g.track_length, 99.5)), GRADES["A"][0] + 2)
    e_hi = max(float(np.percentile(g.reproj_error_px, 99.8)), GRADES["B"][2] * 1.1)
    a_hi = float(np.percentile(g.max_tri_angle_deg, 99.8))
    h_lo, h_hi = np.percentile(g.height_km, [0.5, 99.5])
    nA, nB = GRADES["A"], GRADES["B"]
    specs = [
        ("track_length", "Track length (images)", np.arange(2.5, np.floor(t_hi) + 1.5, 1.0),
         [(nB[0] - 0.5, f"B ≥ {nB[0]}"), (nA[0] - 0.5, f"A ≥ {nA[0]}")], True),
        ("reproj_error_px", "Mean reprojection error (px)", np.linspace(0, e_hi, 51),
         [(nA[2], f"A ≤ {nA[2]:g}"), (nB[2], f"B ≤ {nB[2]:g}")], False),
        ("max_tri_angle_deg", "Max triangulation angle (°)", np.arange(0, a_hi + 2, 2.0),
         [(nB[1], f"B ≥ {nB[1]:g}°"), (nA[1], f"A ≥ {nA[1]:g}°")], False),
        ("height_km", "Height above ellipsoid (km)", np.linspace(h_lo, h_hi, 61),
         [(0.0, "ellipsoid")], False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    for ax, (col, title, bins, lines, logy) in zip(axes.flat, specs):
        data = [g.loc[(g.grade == k) & g[col].between(bins[0], bins[-1]), col] for k in grades]
        ax.hist(data, bins=bins, stacked=True, color=[GRADE_COLORS[k] for k in grades],
                edgecolor=SURFACE, linewidth=0.4)
        for i, (x, label) in enumerate(lines):
            ax.axvline(x, color=INK2, lw=0.9, ls="--")
            ax.text(x, 0.99 - 0.08 * i, f" {label}", transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=8, color=INK2)
        if logy:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,g}"))
        ax.set_xlim(bins[0], bins[-1])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("landmarks")
    counts = lm.grade.value_counts()
    fig.legend(handles=[Patch(color=GRADE_COLORS[k], label=f"grade {k}  ({counts.get(k, 0):,})")
                        for k in grades], loc="lower right", bbox_to_anchor=(0.99, 1.0),
               ncol=len(grades))
    _suptitle(fig, "Landmark quality",
              f"{len(g):,} landmarks stacked by grade; a grade needs all three thresholds "
              f"(A: track ≥ {nA[0]}, angle ≥ {nA[1]:g}°, error ≤ {nA[2]:g} px; B: ≥ {nB[0]}, "
              f"≥ {nB[1]:g}°, ≤ {nB[2]:g} px). x axes end at the 99.5–99.8th percentile.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, ws.plots / "09_quality.png")


@_styled
def plot_radius(ws: Workspace, lm: pd.DataFrame) -> Path:
    """Figure 10: landmark height above the ellipsoid against latitude.

    A logarithmic hexbin density, with the median and the 25th/75th percentiles in 5°
    latitude bands (bands with at least 20 landmarks). The plotted range excludes the
    0.25 % most extreme heights and 0.05 % most extreme latitudes on each side.

    Parameters
    ----------
    ws : Workspace
        Output workspace.
    lm : pandas.DataFrame
        Landmarks with ``lat_deg``, ``height_km`` and ``outlier``.

    Returns
    -------
    pathlib.Path
        ``plots/10_radius_vs_latitude.png``.
    """
    g = lm[~lm.outlier]
    lat_lo, lat_hi = np.percentile(g.lat_deg, [0.05, 99.95])
    lat_lo, lat_hi = 5 * np.floor(lat_lo / 5), 5 * np.ceil(lat_hi / 5)
    h_lo, h_hi = np.percentile(g.height_km, [0.25, 99.75])
    pad = 0.05 * (h_hi - h_lo)
    h_lo, h_hi = h_lo - pad, h_hi + pad
    shown = g[g.lat_deg.between(lat_lo, lat_hi) & g.height_km.between(h_lo, h_hi)]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    hb = ax.hexbin(shown.lat_deg, shown.height_km, gridsize=(80, 32),
                   extent=(lat_lo, lat_hi, h_lo, h_hi), cmap=SEQUENTIAL, norm=LogNorm(), mincnt=1,
                   linewidths=0)
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    bins = np.arange(lat_lo, lat_hi + 5, 5)
    band = g[g.lat_deg.between(lat_lo, lat_hi)]
    grp = band.groupby(pd.cut(band.lat_deg, bins), observed=False).height_km
    q = grp.quantile([0.25, 0.5, 0.75]).unstack()
    q[grp.size().to_numpy() < 20] = np.nan
    mid = (bins[:-1] + bins[1:]) / 2
    halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]
    ax.plot(mid, q[0.5].to_numpy(), color=INK, lw=1.6, label="median per 5° band",
            path_effects=halo)
    ax.plot(mid, q[0.25].to_numpy(), color=INK, lw=0.9, ls="--", label="25th / 75th percentile",
            path_effects=halo)
    ax.plot(mid, q[0.75].to_numpy(), color=INK, lw=0.9, ls="--", path_effects=halo)
    ax.set_xlim(lat_lo, lat_hi)
    ax.set_ylim(h_lo, h_hi)
    ax.legend(loc="upper left")
    counts = hb.get_array()
    cb = _colorbar(fig, hb, "landmarks per cell", ax=ax, pad=0.01)
    _log_ticks(cb, float(counts.min()), float(counts.max()))
    ax.set_xlabel("planetocentric latitude (°)")
    ax.set_ylabel("height above ellipsoid (km)")
    _suptitle(fig, "Radius residuals vs latitude",
              f"Height of {len(shown) / len(g):.1%} of {len(g):,} landmarks above the reference "
              "ellipsoid (0 km line); the low southern latitudes lie in the Rheasilvia basin")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, ws.plots / "10_radius_vs_latitude.png")


@_styled
def plot_chips(ws: Workspace, curated: pd.DataFrame, obs: pd.DataFrame, cams: pd.DataFrame,
               n_landmarks: int = 8, n_views: int = 7, half: int = 24) -> Path:
    """Figure 11: image chips of a few grade-A landmarks across their views.

    Landmarks are picked from the curated set (grade A when possible), spread in
    longitude and preferring long tracks; each row shows ``n_views`` chips from the
    earliest to the latest observation, with the measured keypoint at the centre.

    Parameters
    ----------
    ws : Workspace
        Workspace with prepared PNGs.
    curated : pandas.DataFrame
        Catalog ``landmarks_curated.csv``.
    obs : pandas.DataFrame
        Catalog ``observations.csv``.
    cams : pandas.DataFrame
        Catalog ``cameras.csv`` (image times).
    n_landmarks : int
        Number of rows.
    n_views : int
        Chips per row.
    half : int
        Chip half-width in pixels (chips are ``2 * half + 1`` square).

    Returns
    -------
    pathlib.Path
        ``plots/11_landmark_chips.png``.
    """
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
    W, H = n_views * 1.35 + 1.5, len(picks) * 1.5 + 0.6
    fig, axes = plt.subplots(len(picks), n_views, figsize=(W, H))
    fig.subplots_adjust(left=1.5 / W, right=0.995, bottom=0.01, top=1 - 0.6 / H, wspace=0.05,
                        hspace=0.32)
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
        row[0].text(-0.08, 0.5, f"{lmk.landmark_id}\n{lmk.lat_deg:+.1f}°, {lmk.lon_deg:.1f}°E\n"
                    f"{lmk.track_length} views, grade {lmk.grade}", transform=row[0].transAxes,
                    ha="right", va="center", fontsize=7.5, color=INK)
    _suptitle(fig, "Landmark chips", f"{2 * half + 1}×{2 * half + 1} px around each observation, "
              "earliest to latest view (+ = measured keypoint, title = day and UTC time)")
    return _save(fig, ws.plots / "11_landmark_chips.png")


def interactive_figure(lm: pd.DataFrame, max_points: int = 80000, title: str | None = None):
    """Build a rotatable 3-D plotly scatter of the landmarks, coloured by height.

    Parameters
    ----------
    lm : pandas.DataFrame
        Landmarks (all or curated) with positions, ``height_km``, ``lat_deg``, ``lon_deg``,
        ``track_length``, ``grade``, ``landmark_id`` and ``outlier``.
    max_points : int
        Random subsample size; the browser slows down above about 100k points.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    import plotly.graph_objects as go

    g = lm[~lm.outlier]
    if len(g) > max_points:
        g = g.sample(max_points, random_state=0)
    lim = float(np.nanpercentile(np.abs(g.height_km), 98))
    colorscale = [[i / (len(DIVERGING_STOPS) - 1), c] for i, c in enumerate(DIVERGING_STOPS)]
    hover = (g.landmark_id + "<br>lat " + g.lat_deg.round(2).astype(str) + "°, lon "
             + g.lon_deg.round(2).astype(str) + "°E<br>h " + g.height_km.round(2).astype(str)
             + " km<br>track " + g.track_length.astype(str) + ", grade " + g.grade)
    fig = go.Figure(go.Scatter3d(
        x=g.x_km, y=g.y_km, z=g.z_km, mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=1.6, color=g.height_km, colorscale=colorscale, cmin=-lim, cmax=lim,
                    colorbar=dict(title="height (km)"))))
    fig.update_layout(
        title=title or f"Vesta landmark catalog ({len(g):,} landmarks, body-fixed km)",
        paper_bgcolor=SURFACE,
        scene=dict(aspectmode="data", xaxis_title="x (km)", yaxis_title="y (km)", zaxis_title="z (km)"),
        margin=dict(l=0, r=0, t=40, b=0), font=dict(color=INK))
    return fig


def plot_interactive(ws: Workspace, lm: pd.DataFrame, max_points: int = 80000) -> Path | None:
    """Write ``plots/pointcloud.html``, the interactive version of figure 07.

    Parameters
    ----------
    ws : Workspace
        Output workspace.
    lm : pandas.DataFrame
        Landmarks, as for :func:`interactive_figure`.
    max_points : int
        Random subsample size.

    Returns
    -------
    pathlib.Path or None
        The HTML path, or ``None`` when plotly is not installed. The page loads plotly.js
        from its CDN, so viewing it needs a network connection.
    """
    try:
        fig = interactive_figure(lm, max_points)
    except ImportError:
        log.info("plotly not installed - skipping the interactive page (pip install plotly)")
        return None
    path = ws.plots / "pointcloud.html"
    fig.write_html(path, include_plotlyjs="cdn")
    log.info("wrote %s", path)
    return path


@_styled
def make_all(ws: Workspace, ds: Dataset, interactive: bool = True) -> list[Path]:
    """Draw every figure the workspace has data for.

    Geometry needs only ``metadata.csv``; the montage and orientation check need
    ``prepare.json``; figures 04-11 need the catalog.

    Parameters
    ----------
    ws : Workspace
        Pipeline workspace.
    ds : Dataset
        Dataset definition (camera, body radii).
    interactive : bool
        Also write ``pointcloud.html`` when plotly is available.

    Returns
    -------
    list of pathlib.Path
        Files written, in figure order.
    """
    ws.plots.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(ws.metadata_csv)
    colors = _seq_colors(dict.fromkeys(meta.sequence))
    night = _polar_night(meta["subsolar_lat_deg"]) if "subsolar_lat_deg" in meta else None
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
            plot_pointcloud(ws, lm, ds.body.radii_km),
            plot_map(ws, lm, curated, summary.get("curated_spacing_km"), night),
            plot_quality(ws, lm),
            plot_radius(ws, lm),
            plot_chips(ws, curated, obs, cams),
        ]
        if interactive:
            out.append(plot_interactive(ws, lm))
    return [p for p in out if p is not None]


# ---------------------------------------------------------------- navigation (NCC matching)

def _show_image(ax, img: np.ndarray, origin=(0, 0), alpha: float = 1.0, lo: float = 1.0,
                hi: float = 99.8):
    """Draw an image (or a crop) in COLMAP pixel coordinates with a percentile stretch.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    img : numpy.ndarray
        Image in the package's orientation; NaN pixels are drawn in the surface colour.
    origin : tuple of int
        ``(column, row)`` array index of the top-left pixel, so a crop lands at its place.
    alpha : float
        Opacity; below 1 the image recedes behind the marks drawn over it.
    lo, hi : float
        Percentiles of the finite pixels mapped to black and white.
    """
    finite = img[np.isfinite(img)]
    vmin, vmax = np.percentile(finite, [lo, hi]) if finite.size else (0.0, 1.0)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(SURFACE)
    c0, r0 = origin
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation="nearest",
              extent=(c0, c0 + img.shape[1], r0 + img.shape[0], r0))
    ax.grid(False)


def _image_axes(ax, camera, title: str):
    """Frame an axes on the full image: pixel limits, equal aspect, v down."""
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(title)


def _thin(u, v, cell: float) -> np.ndarray:
    """Indices that keep at most one point per ``cell`` x ``cell`` pixel square."""
    key = np.floor(np.asarray(u) / cell) * 1e4 + np.floor(np.asarray(v) / cell)
    return np.unique(key, return_index=True)[1]


def _nice(x: float) -> float:
    """Round a positive number down to 1, 2 or 5 times a power of ten (for quiver keys)."""
    e = np.floor(np.log10(x))
    m = x / 10 ** e
    return float((5 if m >= 5 else 2 if m >= 2 else 1) * 10 ** e)


@_styled
def plot_nav_matches(out_dir: Path, img: np.ndarray, matches: pd.DataFrame, name: str, camera,
                     min_ncc: float = 0.8, zoom_px: int = 128) -> Path:
    """Navigation figure 01: the landmarks found in one new image.

    The left panel shows the whole image with every inlier coloured by its NCC peak and the
    rejected candidates in grey; the right panel zooms on the densest area and draws, at
    true scale, the shift from the a priori prediction to the NCC measurement.

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory.
    img : numpy.ndarray
        The image, in the package's orientation.
    matches : pandas.DataFrame
        :func:`~asteroid_colmap.ncc.match_image` rows of this image.
    name : str
        Image name (titles).
    camera : FramingCamera
        Camera model (image size).
    min_ncc : float
        Lower end of the NCC colour scale (the acceptance threshold).
    zoom_px : int
        Width of the zoom window (px).

    Returns
    -------
    pathlib.Path
        ``01_matches.png``.
    """
    inl = matches[matches.inlier]
    rej = matches[~matches.inlier]
    ru = rej.u.where(np.isfinite(rej.u), rej.u_apriori)
    rv = rej.v.where(np.isfinite(rej.v), rej.v_apriori)
    fig, (ax, az) = plt.subplots(1, 2, figsize=(12.4, 6.6), width_ratios=(1.18, 1))
    _show_image(ax, img, alpha=0.55)
    ax.scatter(ru, rv, s=9, color=MUTED, linewidths=0.3, edgecolors=SURFACE,
               label="rejected or outlier")
    sc = ax.scatter(inl.u, inl.v, c=inl.ncc, cmap=SEQUENTIAL_MARKS, vmin=min_ncc, vmax=1.0,
                    s=9, linewidths=0.3, edgecolors=SURFACE, label="inlier (colour = NCC)")
    _image_axes(ax, camera, "All searched landmarks")
    _colorbar(fig, sc, "NCC peak", ax=ax, pad=0.015, fraction=0.04)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, markerscale=1.8)

    # zoom on the zoom_px window with the most inliers
    half = zoom_px / 2
    if len(inl):
        cu = np.clip(inl.u.to_numpy(), half, camera.width - half)
        cv = np.clip(inl.v.to_numpy(), half, camera.height - half)
        near = (np.abs(cu[:, None] - cu[None, :]) < half) & (np.abs(cv[:, None] - cv[None, :]) < half)
        k = int(np.argmax(near.sum(axis=1)))
        u0, v0 = cu[k] - half, cv[k] - half
    else:
        u0, v0 = camera.width / 2 - half, camera.height / 2 - half
    c0, r0 = int(np.floor(u0)), int(np.floor(v0))
    _show_image(az, img[r0:r0 + zoom_px, c0:c0 + zoom_px], origin=(c0, r0), alpha=0.55)
    ax.add_patch(plt.Rectangle((c0, r0), zoom_px, zoom_px, fill=False, edgecolor=INK, lw=1.0))
    box = inl[(inl.u >= c0) & (inl.u <= c0 + zoom_px) & (inl.v >= r0) & (inl.v <= r0 + zoom_px)]
    for _, m in box.iterrows():
        az.annotate("", xy=(m.u, m.v), xytext=(m.u_apriori, m.v_apriori),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=3, shrinkB=3,
                                    mutation_scale=7))
    az.scatter(box.u_apriori, box.v_apriori, s=30, color=CATEGORICAL[1], edgecolors=SURFACE,
               linewidths=0.8, label="a priori prediction (label pose)", zorder=3)
    az.scatter(box.u, box.v, s=30, color=CATEGORICAL[0], edgecolors=SURFACE, linewidths=0.8,
               label="NCC measurement", zorder=3)
    az.set_xlim(c0, c0 + zoom_px)
    az.set_ylim(r0 + zoom_px, r0)
    az.set_aspect("equal")
    az.set_xlabel("u (px)")
    az.set_title(f"Zoom: {len(box)} inliers, true scale")
    az.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=1)
    shift = np.hypot(inl.u - inl.u_apriori, inl.v - inl.v_apriori)
    _suptitle(fig, f"Catalog landmarks found in {name}",
              f"{len(inl):,} inliers of {len(matches):,} landmarks searched; median NCC "
              f"{inl.ncc.median():.3f}; the label pose predicts them {shift.median():.1f} px "
              "(median) from where NCC finds them")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=1 - 0.55 / fig.get_figheight(),
                        wspace=0.24)
    return _save(fig, out_dir / "01_matches.png")


@_styled
def plot_nav_chips(out_dir: Path, img: np.ndarray, matches: pd.DataFrame, pose, t, camera,
                   n: int = 8, half: int = 16) -> Path:
    """Navigation figure 02: maplet, rendered template and image for a few landmarks.

    Columns are inliers spread from the lowest to the highest NCC. Rows: the stored maplet
    of the reference view (maplet grid, not image orientation), the maplet rendered into
    the image with the estimated pose, the image over the same window, and the image around
    the a priori prediction with the prediction (+) and the measurement (o) marked.

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory.
    img : numpy.ndarray
        The image, in the package's orientation.
    matches : pandas.DataFrame
        :func:`~asteroid_colmap.ncc.match_image` rows of this image.
    pose : pandas.Series
        The image's ``poses.csv`` row (estimated attitude ``qw..qz`` and position).
    t : Templates
        Maplets used for matching.
    camera : FramingCamera
        Camera model.
    n : int
        Number of landmarks (columns).
    half : int
        Window half-width (px).

    Returns
    -------
    pathlib.Path
        ``02_chips.png``.
    """
    from .geometry import quat_to_matrix
    from .ncc import render_templates

    W = 2 * half + 1
    inl = matches[matches.inlier].assign(u_est=lambda d: d.u + d.residual_u_px,
                                         v_est=lambda d: d.v + d.residual_v_px)

    def inside(u, v):
        """Window centred on (u, v) lies inside the image."""
        c, r = np.round(u - 0.5), np.round(v - 0.5)
        return (c >= half) & (c < camera.width - half) & (r >= half) & (r < camera.height - half)

    inl = inl[inside(inl.u_est, inl.v_est) & inside(inl.u_apriori, inl.v_apriori)].sort_values("ncc")
    pick = inl.iloc[np.unique(np.linspace(0, len(inl) - 1, min(n, len(inl))).round().astype(int))]
    pick = pick.iloc[::-1]
    lid = pd.Series(np.arange(len(t)), index=t.landmark_id)
    idx = lid.loc[pick.landmark_id].to_numpy()
    view = np.array([np.flatnonzero(t.view_image[i] == vi)[0] for i, vi in zip(idx, pick.view_image)])
    R = quat_to_matrix([pose.qw, pose.qx, pose.qy, pose.qz])
    C = np.array([pose.x_km, pose.y_km, pose.z_km])
    col0 = (np.round(pick.u_est - 0.5) - half).astype(int).to_numpy()
    row0 = (np.round(pick.v_est - 0.5) - half).astype(int).to_numpy()
    rendered, _ = render_templates(t, idx, view, R, C, camera, col0, row0, W)

    labels = ["maplet\n(reference view,\nmaplet grid)", "rendered with\nestimated pose",
              "image at\nestimated pose", "image at\na priori pose"]
    Wf, Hf = len(pick) * 1.3 + 1.4, 4 * 1.42 + 0.75
    fig, axes = plt.subplots(4, max(len(pick), 1), figsize=(Wf, Hf), squeeze=False)
    fig.subplots_adjust(left=1.4 / Wf, right=0.995, bottom=0.01, top=1 - 0.95 / Hf, wspace=0.06,
                        hspace=0.1)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    gray = plt.get_cmap("gray").copy()
    gray.set_bad(SURFACE)
    for k, (_, m) in enumerate(pick.iterrows()):
        maplet = t.maplet[idx[k], view[k]].astype(np.float32)
        axes[0, k].imshow(maplet, cmap=gray, vmin=np.nanpercentile(maplet, 1),
                          vmax=np.nanpercentile(maplet, 99.5), interpolation="nearest")
        rd = rendered[k]
        if np.isfinite(rd).any():
            axes[1, k].imshow(rd, cmap=gray, vmin=np.nanpercentile(rd, 1),
                              vmax=np.nanpercentile(rd, 99.5), interpolation="nearest")
        c, r = col0[k], row0[k]
        _show_image(axes[2, k], img[r:r + W, c:c + W], origin=(c, r), hi=99.5)
        ca = int(np.round(m.u_apriori - 0.5)) - half
        ra = int(np.round(m.v_apriori - 0.5)) - half
        _show_image(axes[3, k], img[ra:ra + W, ca:ca + W], origin=(ca, ra), hi=99.5)
        axes[3, k].plot(m.u_apriori, m.v_apriori, "+", color=CATEGORICAL[1], ms=10, mew=1.6)
        axes[3, k].plot(m.u, m.v, "o", mfc="none", mec=CATEGORICAL[0], ms=8, mew=1.6)
        axes[0, k].set_title(f"{m.landmark_id}\nNCC {m.ncc:.3f}", fontsize=7.5, color=INK2,
                             loc="center", weight="normal")
    for r, text in enumerate(labels):
        axes[r, 0].text(-0.1, 0.5, text, transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=8, color=INK)
    fig.legend(handles=[Line2D([], [], color=CATEGORICAL[1], marker="+", ls="", ms=9, mew=1.6,
                               label="a priori prediction"),
                        Line2D([], [], color=CATEGORICAL[0], marker="o", mfc="none", ls="", ms=7,
                               mew=1.6, label="NCC measurement")],
               loc="lower right", bbox_to_anchor=(0.995, 1 - 0.62 / Hf), ncol=2)
    _suptitle(fig, "What NCC compares",
              f"{W}×{W} px windows; the maplets are {t.size}×{t.size} samples at "
              f"{t.spacing_km:.3f} km, each landmark's view chosen for the closest Sun direction")
    return _save(fig, out_dir / "02_chips.png")


@_styled
def plot_nav_residuals(out_dir: Path, matches: pd.DataFrame, name: str, camera,
                       cell_px: float = 36.0) -> Path:
    """Navigation figure 03: pointing error before and residuals after the pose fit.

    Arrows point from the prediction to the NCC measurement over the image frame: from the
    a priori (label) pose in the left panel, from the estimated pose in the middle one, each
    with its own exaggeration (see the keys). The right panel pools the post-fit residuals
    of every image.

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory.
    matches : pandas.DataFrame
        :func:`~asteroid_colmap.ncc.match_images` rows of all images.
    name : str
        Image shown in the two quiver panels.
    camera : FramingCamera
        Camera model (image size).
    cell_px : float
        At most one arrow per ``cell_px`` square, so the field stays readable.

    Returns
    -------
    pathlib.Path
        ``03_residuals.png``.
    """
    allin = matches[matches.inlier]
    m = allin[allin.image == name]
    keep = m.iloc[_thin(m.u, m.v, cell_px)]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), width_ratios=(1, 1, 1.12))
    panels = [  # (axes, du, dv, colour, title, median arrow length as a fraction of the frame)
        (axes[0], keep.u - keep.u_apriori, keep.v - keep.v_apriori, CATEGORICAL[1],
         "Measured − a priori prediction", 0.07),
        (axes[1], -keep.residual_u_px, -keep.residual_v_px, CATEGORICAL[0],
         "Measured − estimated-pose prediction", 0.035),
    ]
    for ax, du, dv, color, title, frac in panels:
        mag = float(np.median(np.hypot(du, dv))) if len(du) else 1.0
        gain = frac * camera.width / max(mag, 1e-3)
        q = ax.quiver(keep.u, keep.v, du, dv, color=color, angles="xy", scale_units="xy",
                      scale=1 / gain, width=0.004, headwidth=3.5, headlength=4)
        key = _nice(2 * mag)
        _image_axes(ax, camera, title)
        ax.set_title(title, pad=20)
        ax.quiverkey(q, 0.0, 1.03, key, f"{key:g} px, arrows drawn ×{gain:.0f}", labelpos="E",
                     coordinates="axes", color=color, labelcolor=INK2,
                     fontproperties={"size": 8.5})
    axes[1].set_ylabel("")

    ax = axes[2]
    du, dv = -allin.residual_u_px.to_numpy(), -allin.residual_v_px.to_numpy()
    lim = max(1.0, float(np.ceil(np.percentile(np.abs(np.r_[du, dv]), 99.5) * 2) / 2)) if len(du) else 1.0
    hb = ax.hexbin(du, dv, gridsize=45, extent=(-lim, lim, -lim, lim), mincnt=1, bins="log",
                   cmap=SEQUENTIAL_MARKS, linewidths=0.2, edgecolors=SURFACE)
    counts = hb.get_array()
    if counts.size:
        cb = _colorbar(fig, hb, "inliers per cell", ax=ax, pad=0.02, fraction=0.045)
        _log_ticks(cb, float(counts.min()), float(counts.max()))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    ax.set_xlabel("u residual (px)")
    ax.set_ylabel("v residual (px)")
    ax.set_title(f"Post-fit residuals, {allin.image.nunique()} images")
    if len(du):
        ax.text(0.03, 0.97, f"RMS u {np.sqrt(np.mean(du**2)):.2f} px, v {np.sqrt(np.mean(dv**2)):.2f} px\n"
                f"median |r| {np.median(np.hypot(du, dv)):.2f} px\n{len(du):,} inliers",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2))
    _suptitle(fig, "Pointing error and fit residuals",
              f"Arrows for {name} (one per {cell_px:.0f} px cell, exaggerated, see keys): a common "
              "shift means a pointing error, a rotation about the centre a twist error")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=1 - 0.75 / fig.get_figheight(),
                        wspace=0.2)
    return _save(fig, out_dir / "03_residuals.png")


def _gap_nan(x: np.ndarray, *ys, factor: float = 5.0):
    """Insert NaN rows where ``x`` jumps by more than ``factor`` times its median step.

    Parameters
    ----------
    x : numpy.ndarray
        Sorted abscissa.
    *ys : numpy.ndarray
        Ordinates aligned with ``x``.
    factor : float
        Jump, in median steps, that breaks the line.

    Returns
    -------
    tuple of numpy.ndarray
        ``x`` and each ``y`` with NaN inserted at the gaps, so separate sequences are not
        joined by a line.
    """
    x = np.asarray(x, float)
    out = [np.asarray(y, float) for y in ys]
    if len(x) < 3:
        return (x, *out)
    step = np.diff(x)
    cut = np.flatnonzero(step > factor * np.median(step)) + 1
    return (np.insert(x, cut, np.nan), *[np.insert(y, cut, np.nan) for y in out])


@_styled
def plot_nav_timeline(out_dir: Path, poses: pd.DataFrame) -> Path:
    """Navigation figure 04: pose correction, residuals, match counts and NCC per image.

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory.
    poses : pandas.DataFrame
        ``poses.csv`` from :func:`~asteroid_colmap.ncc.match_images`.

    Returns
    -------
    pathlib.Path
        ``04_navigation.png``.
    """
    p = poses.assign(time=pd.to_datetime(poses.utc)).sort_values("time")
    hours = ((p.time - p.time.min()).dt.total_seconds() / 3600).to_numpy()
    ok = p.success.astype(bool).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True)
    style = dict(marker="o", ms=3.5, lw=1.3)

    def line(ax, y, color, label, mask=None):
        """Plot one series against time, broken at gaps between sequences."""
        mask = np.ones(len(p), bool) if mask is None else mask
        x, yy = _gap_nan(hours[mask], np.asarray(y, float)[mask])
        ax.plot(x, yy, color=color, label=label, **style)

    ax = axes[0, 0]
    line(ax, p.du_px, CATEGORICAL[0], "u offset (du)", ok)
    line(ax, p.dv_px, CATEGORICAL[1], "v offset (dv)", ok)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("boresight shift (px)")
    ax.set_title("Pointing correction (estimated − label)")
    ax.legend(loc="best")

    ax = axes[0, 1]
    line(ax, p.prefit_rms_px, CATEGORICAL[1], "a priori (label pose)", ok)
    line(ax, p.postfit_rms_px, CATEGORICAL[0], "post-fit (estimated pose)", ok)
    if "sfm_estimate_rms_px" in p and p.sfm_estimate_rms_px.notna().any():
        line(ax, p.sfm_estimate_rms_px, CATEGORICAL[2], "estimated vs catalog pose", ok)
    _log_axis(ax)
    ax.set_ylabel("RMS (px)")
    ax.set_title("Landmark reprojection RMS")
    ax.legend(loc="best")

    ax = axes[1, 0]
    line(ax, p.num_searched, MUTED, "searched")
    line(ax, p.num_accepted, CATEGORICAL[2], "accepted (NCC and margin)")
    line(ax, p.num_inliers, CATEGORICAL[0], "inliers of the pose fit")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("landmarks per image")
    ax.set_title("Matches")
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    line(ax, p.median_ncc, CATEGORICAL[0], None, ok)
    ax.set_ylabel("median NCC of the inliers")
    ax.set_title("Match quality")
    for ax in axes[1]:
        ax.set_xlabel("hours since the first image")
    seqs = ", ".join(dict.fromkeys(p.sequence.astype(str)))
    _suptitle(fig, "Navigation with catalog landmarks",
              f"{ok.sum()} of {len(p)} images with a pose fit ({seqs}, from "
              f"{p.time.min():%d %b %Y %H:%M} UTC)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir / "04_navigation.png")


def _log_axis(ax):
    """Log y axis with plain-number ticks at 1-2-5 steps and no minor labels."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([v for v in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5,
                                                         10, 20, 50, 100, 200, 500, 1000)
                                             if lo <= v <= hi]))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


@_styled
def plot_relative_pose(out_dir: Path, rel: pd.DataFrame, traj: pd.DataFrame | None = None,
                       target: str = "the body") -> Path:
    """Navigation figure 05: relative pose from the landmark matches, against the references.

    Top row: the solved position minus the label position along the range and across it
    (east, north at the sub-spacecraft point), with 1-sigma bars, the smoothed arc of
    :func:`~asteroid_colmap.pose.fit_trajectory` and, for catalog images, the COLMAP position.
    Bottom row: attitude minus the label, landmark reprojection RMS of every pose, and the
    scatter about the arc next to the formal 1-sigma (the covariance check).

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory.
    rel : pandas.DataFrame
        Output of :func:`~asteroid_colmap.pose.relative_poses`.
    traj : pandas.DataFrame, optional
        Output of :func:`~asteroid_colmap.pose.fit_trajectory`.
    target : str
        Body name for the title.

    Returns
    -------
    pathlib.Path
        ``05_relative_pose.png``.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = rel[rel.success.astype(bool)].assign(time=lambda d: pd.to_datetime(d.utc))
    p = p.sort_values("time").reset_index(drop=True)
    t0 = p.time.min()
    hours = ((p.time - t0).dt.total_seconds() / 3600).to_numpy()
    has_sfm = "range_minus_sfm_km" in p and p.range_minus_sfm_km.notna().any()
    if traj is not None and len(traj):
        tr = traj.assign(time=pd.to_datetime(traj.utc)).sort_values("time")
        tr_hours = ((tr.time - t0).dt.total_seconds() / 3600).to_numpy()
    else:
        tr = None
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.8))
    dot = dict(marker="o", ms=4, lw=0, mec=SURFACE, mew=0.4)

    def gap_line(ax, x, y, **kw):
        """Line broken at gaps between sequences."""
        xx, yy = _gap_nan(x, y)
        ax.plot(xx, yy, **kw)

    for ax, k, name in zip(axes[0], ("range", "east", "north"), ("Range", "East", "North")):
        ax.axhline(0, color=AXIS, lw=0.8)
        if tr is not None:
            mid, sig = tr[f"{k}_minus_label_km"].to_numpy(), tr[f"sigma_{k}_km"].to_numpy()
            xx, lo, hi = _gap_nan(tr_hours, mid - sig, mid + sig)
            ax.fill_between(xx, lo, hi, color=GRID, lw=0, label="smoothed arc ±1σ")
            gap_line(ax, tr_hours, mid, color=INK2, lw=1.4, label="smoothed arc")
        ax.errorbar(hours, p[f"{k}_minus_label_km"], yerr=p[f"sigma_{k}_km"], color=CATEGORICAL[0],
                    elinewidth=0.9, capsize=0, label="PnP, ±1σ", **dot)
        if has_sfm:
            ax.plot(hours, p[f"{k}_minus_label_km"] - p[f"{k}_minus_sfm_km"],
                    color=CATEGORICAL[2], label="COLMAP (catalog image)", **dot)
        ax.set_ylabel("km")
        ax.set_title(f"{name}: estimate − label position")
    axes[0, 0].legend(loc="best")

    ax = axes[1, 0]
    ax.plot(hours, p.attitude_minus_label_arcsec, color=CATEGORICAL[0], label="PnP", **dot)
    if "ncc_attitude_minus_label_arcsec" in p and p.ncc_attitude_minus_label_arcsec.notna().any():
        ax.plot(hours, p.ncc_attitude_minus_label_arcsec, color=MUTED, marker="s", ms=3.5, lw=0,
                label="attitude-only fit (label position)")
    if has_sfm:
        ax.plot(hours, p.sfm_attitude_minus_label_arcsec, color=CATEGORICAL[2],
                label="COLMAP (catalog image)", **dot)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("rotation angle (arcsec)")
    ax.set_title("Attitude: estimate − label")
    ax.legend(loc="best")

    ax = axes[1, 1]
    series = [("label_rms_px", CATEGORICAL[1], "o", "label pose"),
              ("ncc_rms_px", MUTED, "s", "attitude-only fit"),
              ("rms_px", CATEGORICAL[0], "o", "PnP"),
              ("sfm_rms_px", CATEGORICAL[2], "o", "COLMAP (catalog image)")]
    for col, color, marker, label in series:
        if col in p and p[col].notna().any():
            ax.plot(hours, p[col], color=color, marker=marker, ms=3.5, lw=0, mec=SURFACE, mew=0.4,
                    label=label)
    _log_axis(ax)
    ax.set_ylabel("RMS (px)")
    ax.set_title("Landmark reprojection RMS, same inliers")
    ax.legend(loc="best")

    ax = axes[1, 2]
    axes_names = ["range", "east", "north"]
    x = np.arange(3)
    formal = [p[f"sigma_{k}_km"].median() for k in axes_names]
    if tr is not None:
        scatter = [np.sqrt(np.mean(tr[f"residual_{k}_km"] ** 2)) for k in axes_names]
        ax.bar(x, scatter, width=0.55, color=CATEGORICAL[0], label="scatter about the arc (RMS)")
        chi2 = ", ".join(f"{s} {v:.2f}" for s, v in tr.groupby("sequence").chi2_dof.first().items())
        ax.set_title(f"Uncertainty check (χ²/dof: {chi2})", fontsize=10.5)
    else:
        ax.set_title("Uncertainty check")
        ax.text(0.5, 0.82, "no arc: it needs three or more images per sequence", ha="center",
                va="center", color=INK2, transform=ax.transAxes)
    ax.plot(x, formal, color=INK, marker="_", ms=26, mew=2, lw=0, label="formal 1σ (median)")
    ax.set_xticks(x, ["range", "east", "north"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, max(formal + (scatter if tr is not None else [])) * (1.35 if tr is not None
                                                                         else 1.6))
    ax.set_ylabel("km")
    ax.legend(loc="upper left")

    for ax in [*axes[0], axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("hours since the first image")
    tgt = np.hypot(p.sigma_target_u_px, p.sigma_target_v_px).median()
    _suptitle(fig, f"Relative pose with respect to {target} (PnP on the landmark matches)",
              f"{len(p)} of {len(rel)} images solved without the label pose; median range "
              f"{p.range_km.median():.0f} km, formal 1σ {p.sigma_range_km.median():.2f} km along "
              f"the range, {np.median(formal[1:]):.2f} km across it, {tgt:.2f} px for the body "
              f"centre in the image")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir / "05_relative_pose.png")


def make_navigation_plots(out_dir: str | Path, meta: pd.DataFrame, matches: pd.DataFrame,
                          poses: pd.DataFrame, t, camera, flip: str | None = None,
                          image: str | None = None, relative: pd.DataFrame | None = None,
                          trajectory: pd.DataFrame | None = None,
                          target: str = "the body") -> list[Path]:
    """Draw the navigation figures of a matching run (05 only with ``relative``).

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Output directory (created).
    meta : pandas.DataFrame
        Metadata of the matched images (needs ``image`` and ``fit_path``).
    matches, poses : pandas.DataFrame
        Output of :func:`~asteroid_colmap.ncc.match_images`.
    t : Templates
        Maplets used for matching.
    camera : FramingCamera
        Camera model.
    flip : str, optional
        Image flip; defaults to the templates'.
    image : str, optional
        Image for figures 01-03; defaults to the one with the median number of inliers.
    relative, trajectory : pandas.DataFrame, optional
        Output of :func:`~asteroid_colmap.pose.relative_poses` and
        :func:`~asteroid_colmap.pose.fit_trajectory`, for figure 05.
    target : str
        Body name for the titles.

    Returns
    -------
    list of pathlib.Path
        Files written.
    """
    from .preprocess import apply_flip, load_fits

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = poses[poses.success.astype(bool)]
    if not len(ok):
        log.warning("no image has a pose fit - skipping the navigation figures")
        return []
    if image is None:
        image = ok.iloc[(ok.num_inliers - ok.num_inliers.median()).abs().argsort().iloc[0]].image
    pose = ok.set_index("image").loc[image]
    row = meta.set_index("image").loc[image]
    img = apply_flip(load_fits(row["fit_path"]), flip or t.flip)
    mt = matches[matches.image == image]
    min_ncc = float(mt.ncc[mt.accepted].min()) if mt.accepted.any() else 0.8
    out = [plot_nav_matches(out_dir, img, mt, image, camera, min_ncc=np.floor(min_ncc * 20) / 20),
           plot_nav_chips(out_dir, img, mt, pose, t, camera),
           plot_nav_residuals(out_dir, matches, image, camera),
           plot_nav_timeline(out_dir, poses)]
    if relative is not None and relative.success.astype(bool).any():
        out.append(plot_relative_pose(out_dir, relative, trajectory, target))
    return out
