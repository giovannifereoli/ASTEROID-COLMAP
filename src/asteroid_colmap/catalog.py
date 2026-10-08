"""Landmark catalog: COLMAP 3-D points in body-fixed coordinates with quality metrics.

Outputs (in ``<workdir>/catalog``):

``landmarks.csv``          every triangulated point that passes the minimum track length
``landmarks_curated.csv``  best landmark per equal-area surface cell (grades A/B, no outliers)
``observations.csv``       every image measurement of every landmark, in three pixel frames
``cameras.csv``            per-image geometry, registration status and alignment residuals
``landmarks.ply``          point cloud (km, body-fixed) for MeshLab / CloudCompare
``summary.json``           counts, alignment statistics and the similarity transform
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .camera import FramingCamera
from .config import Body
from .geometry import cart_to_latlonr, ellipsoid_radius
from .georef import Similarity
from .metadata import label_geometry
from .model_io import SparseModel

log = logging.getLogger(__name__)

GRADES = {  # min track length, min triangulation angle (deg), max reprojection error (px)
    "A": (8, 15.0, 1.0),
    "B": (4, 5.0, 2.0),
}


def grade(track, tri_angle, error) -> np.ndarray:
    """Quality grade of every landmark.

    A landmark gets the highest grade in :data:`GRADES` whose three thresholds it meets,
    and ``"C"`` otherwise.

    Parameters
    ----------
    track : numpy.ndarray
        Track lengths (number of images).
    tri_angle : numpy.ndarray
        Maximum triangulation angles (deg).
    error : numpy.ndarray
        Mean reprojection errors (px).

    Returns
    -------
    numpy.ndarray
        ``"A"``, ``"B"`` or ``"C"`` per landmark.
    """
    out = np.full(len(track), "C", dtype="<U1")
    for g in ("B", "A"):
        n, ang, err = GRADES[g]
        out[(track >= n) & (tri_angle >= ang) & (error <= err)] = g
    return out


def triangulation_stats(obs: pd.DataFrame, X: np.ndarray, centers: dict[int, np.ndarray],
                        point_index: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Maximum pairwise ray angle and mean camera range of every point.

    Parameters
    ----------
    obs : pandas.DataFrame
        Track elements with ``point3d_id`` and ``image_id``.
    X : numpy.ndarray
        ``(N, 3)`` points, body-fixed (km).
    centers : dict of int to numpy.ndarray
        Camera centre per ``image_id``, body-fixed (km).
    point_index : dict of int to int
        Row of ``X`` per ``point3d_id``.

    Returns
    -------
    max_angle_deg : numpy.ndarray
        ``(N,)`` largest angle between two viewing rays of each point (deg).
    mean_range_km : numpy.ndarray
        ``(N,)`` mean camera-to-point distance (km).
    """
    max_angle = np.zeros(len(X))
    mean_range = np.zeros(len(X))
    for pid, grp in obs.groupby("point3d_id")["image_id"]:
        i = point_index[pid]
        rays = X[i] - np.stack([centers[c] for c in grp.to_numpy()])
        rng = np.linalg.norm(rays, axis=1)
        u = rays / rng[:, None]
        max_angle[i] = np.degrees(np.arccos(np.clip((u @ u.T).min(), -1.0, 1.0)))
        mean_range[i] = rng.mean()
    return max_angle, mean_range


def pixel_frames(u, v, flip: str, camera: FramingCamera) -> dict[str, np.ndarray]:
    """Convert COLMAP pixel coordinates to IK and stored-FITS coordinates.

    Parameters
    ----------
    u, v : array_like
        COLMAP pixel coordinates (pixel centres at +0.5).
    flip : str
        Flip that was applied to the FITS arrays in :func:`~asteroid_colmap.preprocess.prepare`.
    camera : FramingCamera
        Supplies the image size.

    Returns
    -------
    dict of numpy.ndarray
        ``sample_ik``, ``line_ik``: 0-based IK sample and line (the PNG axes).
        ``fits_col``, ``fits_row``: 0-based column and row of the FITS array as stored.
    """
    s, l = np.asarray(u) - 0.5, np.asarray(v) - 0.5
    col = (camera.width - 1) - s if flip in ("lr", "rot180") else s
    row = (camera.height - 1) - l if flip in ("ud", "rot180") else l
    return {"sample_ik": s, "line_ik": l, "fits_col": col, "fits_row": row}


def equal_area_cells(lat, lon, spacing_km: float, mean_radius_km: float) -> np.ndarray:
    """Cell id on an approximately equal-area latitude/longitude grid.

    Latitude bands are ``spacing_km`` tall; each band is split into as many longitude
    cells as fit its circumference at ``spacing_km`` (at least one).

    Parameters
    ----------
    lat, lon : array_like
        Planetocentric latitude and east longitude in [0, 360) (deg).
    spacing_km : float
        Nominal cell size (km).
    mean_radius_km : float
        Sphere radius used to convert km to degrees.

    Returns
    -------
    numpy.ndarray
        Integer id ``band * 100000 + column``.
    """
    dlat = np.degrees(spacing_km / mean_radius_km)
    band = np.floor((np.asarray(lat) + 90.0) / dlat).astype(int)
    lat_c = np.radians(-90.0 + (band + 0.5) * dlat)
    n_lon = np.maximum(1, np.round(2 * np.pi * mean_radius_km * np.cos(lat_c) / spacing_km))
    col = np.floor(np.asarray(lon) / 360.0 * n_lon).astype(int)
    return band * 100000 + col


def one_observation_per_image(model: SparseModel,
                              camera: FramingCamera) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one observation per point and image: the one closest to the projection.

    COLMAP can put two nearby keypoints of one image, both within the reprojection
    threshold, in the same track. Counting both inflates the track length and gives that
    image two measurements of one landmark, so the one with the smaller reprojection error
    is kept. For the points that lose an observation, ``track_length`` (distinct images)
    and ``error`` (mean reprojection error, px) are recomputed from the kept ones; the
    others keep COLMAP's values.

    Parameters
    ----------
    model : SparseModel
        COLMAP model.
    camera : FramingCamera
        Camera model, used to project the points.

    Returns
    -------
    points, observations : pandas.DataFrame
        Copies of ``model.points`` and ``model.observations``, the observations in their
        original order.
    """
    obs = model.observations
    images = model.images.set_index("image_id")
    xyz = model.points.set_index("point3d_id")[["x", "y", "z"]]
    err = np.empty(len(obs))
    for image_id, idx in obs.groupby("image_id").indices.items():
        grp = obs.iloc[idx]
        R, t = images.loc[image_id, "R"], images.loc[image_id, ["tx", "ty", "tz"]].to_numpy(float)
        uv = camera.project(xyz.loc[grp.point3d_id].to_numpy() @ R.T + t)
        err[idx] = np.hypot(uv[:, 0] - grp.u.to_numpy(), uv[:, 1] - grp.v.to_numpy())
    kept = (obs.assign(err=err).sort_values("err", kind="stable")
            .drop_duplicates(["point3d_id", "image_id"]).sort_index())
    points = model.points.copy()
    changed = points.point3d_id.isin(obs.point3d_id[~obs.index.isin(kept.index)])
    stats = kept[kept.point3d_id.isin(points.point3d_id[changed])].groupby("point3d_id").err
    points.loc[changed, "track_length"] = points.point3d_id[changed].map(stats.size())
    points.loc[changed, "error"] = points.point3d_id[changed].map(stats.mean())
    return points, kept.drop(columns="err")


def build_catalog(
    model: SparseModel,
    sim: Similarity,
    per_image: pd.DataFrame,
    meta: pd.DataFrame,
    body: Body,
    camera: FramingCamera,
    flip: str,
    min_track: int = 3,
    max_abs_height_km: float = 60.0,
    curated_spacing_km: float = 10.0,
) -> dict[str, pd.DataFrame]:
    """Turn a georeferenced COLMAP model into the landmark catalog tables.

    Each point keeps one observation per image (:func:`one_observation_per_image`).
    Points seen in at least ``min_track`` images are mapped to the body-fixed frame,
    graded (:func:`grade`) and flagged as outliers when ``|height| > max_abs_height_km``.
    Landmarks are sorted non-outliers first, then by grade, track length (descending)
    and reprojection error, and numbered ``LMK_000001``, ... in that order.

    Parameters
    ----------
    model : SparseModel
        COLMAP model.
    sim : Similarity
        Model -> body-fixed transform (:func:`~asteroid_colmap.georef.georeference`).
    per_image : pandas.DataFrame
        Per-image alignment residuals from the same call.
    meta : pandas.DataFrame
        Image metadata.
    body : Body
        Target body.
    camera : FramingCamera
        Camera model.
    flip : str
        Flip applied in the prepare step, for the FITS pixel frame.
    min_track : int
        Minimum track length of a landmark.
    max_abs_height_km : float
        Height above the ellipsoid beyond which a landmark is an outlier (km).
    curated_spacing_km : float
        Cell size of the curated subset (km).

    Returns
    -------
    dict of pandas.DataFrame
        ``landmarks``: ``landmark_id``, ``colmap_point_id``, ``x/y/z_km``, ``lat_deg``,
        ``lon_deg``, ``radius_km``, ``ellipsoid_radius_km``, ``height_km``,
        ``track_length``, ``reproj_error_px``, ``max_tri_angle_deg``, ``mean_range_km``,
        ``gsd_km``, ``gray``, ``grade`` and ``outlier``.
        ``curated``: the best non-outlier A/B landmark of each equal-area cell, scored
        by ``track * min(angle, 30 deg) / max(error, 0.2 px)``, same columns.
        ``observations``: see :func:`_observations`.
        ``cameras``: see :func:`_cameras`.
    """
    points, observations = one_observation_per_image(model, camera)
    pts = points[points.track_length >= min_track].reset_index(drop=True)
    obs = observations[observations.point3d_id.isin(pts.point3d_id)].reset_index(drop=True)
    X_sfm = pts[["x", "y", "z"]].to_numpy()
    X = sim.apply(X_sfm)
    lat, lon, r = cart_to_latlonr(X)
    r_ell = ellipsoid_radius(lat, lon, body.radii_km)

    images = model.images.set_index("image_id")
    centers = {i: sim.apply(c) for i, c in images["center"].items()}
    point_index = dict(zip(pts.point3d_id, range(len(pts))))
    tri, rng = triangulation_stats(obs, X, centers, point_index)

    lm = pd.DataFrame({
        "colmap_point_id": pts.point3d_id,
        "x_km": X[:, 0], "y_km": X[:, 1], "z_km": X[:, 2],
        "lat_deg": lat, "lon_deg": lon, "radius_km": r,
        "ellipsoid_radius_km": r_ell, "height_km": r - r_ell,
        "track_length": pts.track_length, "reproj_error_px": pts.error,
        "max_tri_angle_deg": tri, "mean_range_km": rng,
        "gsd_km": rng * camera.ifov_rad,
        "gray": np.round(pts[["r", "g", "b"]].mean(axis=1)).astype(int),
    })
    lm["grade"] = grade(lm.track_length.to_numpy(), tri, lm.reproj_error_px.to_numpy())
    lm["outlier"] = lm.height_km.abs() > max_abs_height_km
    lm = lm.sort_values(["outlier", "grade", "track_length", "reproj_error_px"],
                        ascending=[True, True, False, True]).reset_index(drop=True)
    lm.insert(0, "landmark_id", [f"LMK_{i:06d}" for i in range(1, len(lm) + 1)])

    # curated subset: best-scoring A/B landmark in each equal-area cell
    good = lm[~lm.outlier & lm.grade.isin(["A", "B"])].copy()
    good["cell"] = equal_area_cells(good.lat_deg, good.lon_deg, curated_spacing_km,
                                    float(np.mean(body.radii_km)))
    good["score"] = (good.track_length * np.minimum(good.max_tri_angle_deg, 30.0)
                     / np.maximum(good.reproj_error_px, 0.2))
    # stable sort: a tie goes to the better-ranked (lower) landmark ID
    curated = (good.sort_values("score", ascending=False, kind="stable").drop_duplicates("cell")
               .sort_values("landmark_id").drop(columns=["cell", "score"]).reset_index(drop=True))

    observations = _observations(obs, lm, X_sfm, pts, images, meta, body, camera, flip)
    cameras = _cameras(meta, per_image, observations)
    return {"landmarks": lm, "curated": curated, "observations": observations, "cameras": cameras}


def _observations(obs, lm, X_sfm, pts, images, meta, body, camera, flip) -> pd.DataFrame:
    """One row per landmark measurement, sorted by ``landmark_id`` and ``image``.

    Columns: ``landmark_id``, ``image``, ``u``, ``v`` (COLMAP px); the
    :func:`pixel_frames` columns; ``residual_u/v_px`` (COLMAP projection minus
    measurement); ``label_pred_du/dv_px`` (projection with the label pose minus
    measurement); and ``emission_deg`` / ``incidence_deg`` relative to the ellipsoid
    normal.
    """
    ids = dict(zip(lm.colmap_point_id, lm.landmark_id))
    xyz_bf = lm.set_index("colmap_point_id")[["x_km", "y_km", "z_km"]]
    row_of = dict(zip(pts.point3d_id, range(len(pts))))
    lookup = meta.set_index("image")
    out = []
    for image_id, grp in obs.groupby("image_id"):
        name = images.loc[image_id, "name"]
        R, t = images.loc[image_id, "R"], images.loc[image_id, ["tx", "ty", "tz"]].to_numpy(float)
        Xs = X_sfm[[row_of[p] for p in grp.point3d_id]]
        uv_sfm = camera.project(Xs @ R.T + t)
        Xb = xyz_bf.loc[grp.point3d_id].to_numpy()
        g = label_geometry(lookup.loc[name], body)
        uv_lab = camera.project((Xb - g["sc_bf"]) @ g["R_cam_from_bf"].T)
        normal = Xb / np.square(body.radii_km)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        to_sc = g["sc_bf"] - Xb
        to_sc /= np.linalg.norm(to_sc, axis=1, keepdims=True)
        df = pd.DataFrame({
            "landmark_id": grp.point3d_id.map(ids).to_numpy(),
            "image": name,
            "u": grp.u.to_numpy(), "v": grp.v.to_numpy(),
            **pixel_frames(grp.u.to_numpy(), grp.v.to_numpy(), flip, camera),
            "residual_u_px": uv_sfm[:, 0] - grp.u.to_numpy(),
            "residual_v_px": uv_sfm[:, 1] - grp.v.to_numpy(),
            "label_pred_du_px": uv_lab[:, 0] - grp.u.to_numpy(),
            "label_pred_dv_px": uv_lab[:, 1] - grp.v.to_numpy(),
            "emission_deg": np.degrees(np.arccos(np.clip((normal * to_sc).sum(1), -1, 1))),
            "incidence_deg": np.degrees(np.arccos(np.clip(normal @ g["sun_bf"], -1, 1))),
        })
        out.append(df)
    obs_df = pd.concat(out, ignore_index=True)
    return obs_df.sort_values(["landmark_id", "image"]).reset_index(drop=True)


def _cameras(meta, per_image, observations) -> pd.DataFrame:
    """One row per input image, registered or not.

    Columns: ``image``, ``sequence``, ``utc``, ``registered``, ``observation_id``,
    ``exposure_ms``, ``range_km``, ``pixel_scale_km``, ``subsc_lat_deg``,
    ``subsc_lon_deg`` (from the rotation model), ``phase_deg``,
    ``rotation_model_check_deg``; ``num_landmarks`` (0 if unregistered),
    ``reproj_rms_px``; the median label offsets ``label_offset_u/v_px`` (projection with
    the label pose minus measurement) and their length ``label_offset_px``, i.e. how far
    the label pose places the catalog landmarks from where they are seen; then the
    :func:`~asteroid_colmap.georef.georeference` per-image columns.
    """
    keep = ["image", "sequence", "utc", "observation_id", "exposure_ms", "range_km",
            "pixel_scale_km", "subsc_lat_deg", "subsc_lon_model_deg", "phase_deg",
            "rotation_model_check_deg"]
    cams = meta[keep].rename(columns={"subsc_lon_model_deg": "subsc_lon_deg"})
    obs = observations.assign(r2=observations.residual_u_px**2 + observations.residual_v_px**2)
    stats = obs.groupby("image").agg(
        num_landmarks=("landmark_id", "size"),
        reproj_rms_px=("r2", lambda s: float(np.sqrt(s.mean()))),
        label_offset_u_px=("label_pred_du_px", "median"),
        label_offset_v_px=("label_pred_dv_px", "median"),
    )
    stats["label_offset_px"] = np.hypot(stats.label_offset_u_px, stats.label_offset_v_px)
    cams = cams.merge(stats, left_on="image", right_index=True, how="left")
    cols = [c for c in per_image.columns if c not in ("sequence", "utc", "range_km")]
    cams = cams.merge(per_image[cols], on="image", how="left")
    cams.insert(3, "registered", cams.image_id.notna())
    cams["num_landmarks"] = cams.num_landmarks.fillna(0).astype(int)
    cams["image_id"] = cams.image_id.astype("Int64")
    return cams


def write_ply(path: Path, lm: pd.DataFrame) -> None:
    """Write the landmarks as an ASCII PLY point cloud.

    Each vertex has ``x``, ``y``, ``z`` (km, body-fixed), a grey RGB colour from
    ``gray`` and a ``height_km`` scalar property.

    Parameters
    ----------
    path : pathlib.Path
        Output file.
    lm : pandas.DataFrame
        Landmarks table.
    """
    header = "\n".join([
        "ply", "format ascii 1.0", "comment asteroid-colmap landmarks, body-fixed km",
        f"element vertex {len(lm)}", "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "property float height_km", "end_header",
    ])
    g = lm.gray.clip(0, 255).to_numpy()
    data = np.column_stack([lm.x_km, lm.y_km, lm.z_km, g, g, g, lm.height_km])
    np.savetxt(path, data, fmt="%.5f %.5f %.5f %d %d %d %.4f", header=header, comments="")


def summarize(cat: dict[str, pd.DataFrame], align: dict, sim: Similarity, recon: dict,
              prep: dict, curated_spacing_km: float | None = None) -> dict:
    """Catalog statistics for ``summary.json``.

    Parameters
    ----------
    cat : dict of pandas.DataFrame
        Output of :func:`build_catalog`.
    align : dict
        Alignment summary from :func:`~asteroid_colmap.georef.georeference`.
    sim : Similarity
        Model -> body-fixed transform.
    recon : dict
        Output of :func:`~asteroid_colmap.reconstruct.reconstruct`.
    prep : dict
        Output of :func:`~asteroid_colmap.preprocess.prepare`.
    curated_spacing_km : float, optional
        Recorded as is.

    Returns
    -------
    dict
        ``num_input_images``, ``num_registered_images``, ``registered_by_sequence``,
        ``num_landmarks``, ``num_curated_landmarks``, ``curated_spacing_km``,
        ``num_outliers``, ``grades``, ``num_observations``; medians over non-outliers
        ``median_track_length``, ``median_reproj_error_px``, ``median_tri_angle_deg``;
        ``height_km_p05_p50_p95``, ``lat_range_deg``,
        ``label_pointing_offset_median_px`` (median ``label_offset_px``),
        ``label_pointing_offset_by_sequence_px`` (median ``[u, v]`` label offset of each
        sequence), ``alignment``, ``similarity``,
        ``image_flip``, ``colmap`` and ``selected_model``.
    """
    lm, cams, obs = cat["landmarks"], cat["cameras"], cat["observations"]
    good = lm[~lm.outlier]
    return {
        "num_input_images": int(len(cams)),
        "num_registered_images": int(cams.registered.sum()),
        "registered_by_sequence": cams.groupby("sequence").registered.sum().astype(int).to_dict(),
        "num_landmarks": int(len(lm)),
        "num_curated_landmarks": int(len(cat["curated"])),
        "curated_spacing_km": curated_spacing_km,
        "num_outliers": int(lm.outlier.sum()),
        "grades": lm.grade.value_counts().sort_index().astype(int).to_dict(),
        "num_observations": int(len(obs)),
        "median_track_length": float(good.track_length.median()),
        "median_reproj_error_px": float(good.reproj_error_px.median()),
        "median_tri_angle_deg": float(good.max_tri_angle_deg.median()),
        "height_km_p05_p50_p95": good.height_km.quantile([0.05, 0.5, 0.95]).round(3).tolist(),
        "lat_range_deg": [float(good.lat_deg.min()), float(good.lat_deg.max())],
        "label_pointing_offset_median_px": float(cams.label_offset_px.median()),
        "label_pointing_offset_by_sequence_px": {
            seq: [round(float(g.label_offset_u_px.median()), 3),
                  round(float(g.label_offset_v_px.median()), 3)]
            for seq, g in cams[cams.registered].groupby("sequence")},
        "alignment": align,
        "similarity": sim.as_dict(),
        "image_flip": prep.get("flip"),
        "colmap": recon.get("colmap"),
        "selected_model": recon.get("selected_model"),
    }


def write_catalog(out_dir: Path, cat: dict[str, pd.DataFrame]) -> None:
    """Write the catalog tables and the point cloud.

    Writes ``landmarks.csv``, ``landmarks_curated.csv``, ``observations.csv``,
    ``cameras.csv`` and ``landmarks.ply`` (see the module docstring).

    Parameters
    ----------
    out_dir : pathlib.Path
        Output directory; created if needed.
    cat : dict of pandas.DataFrame
        Output of :func:`build_catalog`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cat["landmarks"].to_csv(out_dir / "landmarks.csv", index=False, float_format="%.6f")
    cat["curated"].to_csv(out_dir / "landmarks_curated.csv", index=False, float_format="%.6f")
    cat["observations"].to_csv(out_dir / "observations.csv", index=False, float_format="%.4f")
    cat["cameras"].to_csv(out_dir / "cameras.csv", index=False, float_format="%.6f")
    write_ply(out_dir / "landmarks.ply", cat["landmarks"])
    log.info("catalog: %d landmarks (%d curated), %d observations -> %s",
             len(cat["landmarks"]), len(cat["curated"]), len(cat["observations"]), out_dir)


def compare_catalogs(dir_a: Path, dir_b: Path, min_shared: int = 3) -> dict:
    """Compare two catalogs built from the same feature database.

    Typical use is the label-poses and incremental modes. Landmarks are matched through
    shared keypoints (same image and ``u, v`` to 0.01 px); each landmark is paired at
    most once, with the partner it shares most keypoints with.

    Parameters
    ----------
    dir_a, dir_b : pathlib.Path
        Catalog directories with ``observations.csv`` and ``landmarks.csv``.
    min_shared : int
        Minimum number of shared keypoints for a match.

    Returns
    -------
    dict
        ``num_landmarks`` (both catalogs) and ``num_matched``; if any matched, also
        ``distance_km_p50_p90_p99``, ``mean_offset_km`` (a - b),
        ``distance_minus_offset_km_p50_p90`` and ``height_difference_km_p50_absp90``.
    """
    cols = ["landmark_id", "image", "u", "v"]
    obs = []
    for d in (dir_a, dir_b):
        o = pd.read_csv(Path(d) / "observations.csv", usecols=cols)
        o["key"] = o.image + ":" + o.u.round(2).astype(str) + ":" + o.v.round(2).astype(str)
        obs.append(o[["landmark_id", "key"]])
    m = obs[0].merge(obs[1], on="key", suffixes=("_a", "_b"))
    pairs = m.groupby(["landmark_id_a", "landmark_id_b"]).size().rename("shared").reset_index()
    pairs = (pairs[pairs.shared >= min_shared].sort_values("shared", ascending=False)
             .drop_duplicates("landmark_id_a").drop_duplicates("landmark_id_b"))
    la = pd.read_csv(Path(dir_a) / "landmarks.csv").set_index("landmark_id")
    lb = pd.read_csv(Path(dir_b) / "landmarks.csv").set_index("landmark_id")
    out = {"num_landmarks": [int(len(la)), int(len(lb))], "num_matched": int(len(pairs))}
    if pairs.empty:
        return out
    xyz = ["x_km", "y_km", "z_km"]
    diff = la.loc[pairs.landmark_id_a, xyz].to_numpy() - lb.loc[pairs.landmark_id_b, xyz].to_numpy()
    dh = (la.loc[pairs.landmark_id_a, "height_km"].to_numpy()
          - lb.loc[pairs.landmark_id_b, "height_km"].to_numpy())
    offset = diff.mean(0)
    return {
        **out,
        "distance_km_p50_p90_p99": np.percentile(np.linalg.norm(diff, axis=1), [50, 90, 99]).tolist(),
        "mean_offset_km": offset.tolist(),
        "distance_minus_offset_km_p50_p90": np.percentile(
            np.linalg.norm(diff - offset, axis=1), [50, 90]).tolist(),
        "height_difference_km_p50_absp90": [float(np.median(dh)), float(np.percentile(np.abs(dh), 90))],
    }
