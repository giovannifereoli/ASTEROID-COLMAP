"""Tie the COLMAP model (arbitrary similarity frame) to the body-fixed frame.

A 7-parameter similarity ``X_bf = s R X_sfm + t`` is fitted between COLMAP camera centres
and the label spacecraft positions. Camera centres on near-planar arcs cannot tell a model
from its mirror image, so the label attitudes are then used as an independent check: after
alignment each COLMAP rotation should match the label camera attitude to within the
pointing knowledge (a mirrored or Necker-reversed model shows up as tens of degrees).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .camera import FramingCamera
from .config import Body
from .geometry import rotation_angle_deg, umeyama
from .metadata import label_geometry
from .model_io import SparseModel

log = logging.getLogger(__name__)


@dataclass
class Similarity:
    """Similarity transform ``X_bf = scale R X_sfm + t`` from the COLMAP to the body frame.

    Attributes
    ----------
    scale : float
        Scale factor (km per COLMAP model unit).
    R : numpy.ndarray
        ``(3, 3)`` proper rotation.
    t : numpy.ndarray
        ``(3,)`` translation (km).
    """
    scale: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Map points from the COLMAP model frame to the body-fixed frame.

        Parameters
        ----------
        X : array_like
            ``(N, 3)`` or ``(3,)`` points in model units.

        Returns
        -------
        numpy.ndarray
            Body-fixed points (km), same shape as ``X``.
        """
        return self.scale * np.asarray(X) @ self.R.T + self.t

    def as_dict(self) -> dict:
        """JSON-ready form of the transform.

        Returns
        -------
        dict
            ``scale_km_per_unit``, ``R`` (nested lists) and ``t_km``.
        """
        return {"scale_km_per_unit": self.scale, "R": self.R.tolist(), "t_km": self.t.tolist()}


def robust_similarity(src, dst, n_iter: int = 5, k_sigma: float = 4.0, floor: float = 1.0):
    """Fit a similarity transform with iterative outlier rejection.

    Each pass fits :func:`~asteroid_colmap.geometry.umeyama` to the current inliers, then
    keeps the points whose residual is at most ``max(median + k_sigma * 1.4826 * MAD,
    floor)``, with the median and MAD of the current inliers. It stops after ``n_iter``
    passes, when the inlier set stops changing, or when fewer than 3 points would remain.

    Parameters
    ----------
    src, dst : numpy.ndarray
        ``(N, 3)`` corresponding points, ``dst ~ s R src + t``.
    n_iter : int
        Maximum number of fit-and-reject passes.
    k_sigma : float
        Rejection threshold in robust standard deviations.
    floor : float
        Lower bound of the rejection threshold, in ``dst`` units (km), so that a very
        tight fit does not reject good points.

    Returns
    -------
    sim : Similarity
        The last fit.
    res : numpy.ndarray
        ``(N,)`` residual norm of every point under ``sim``, in ``dst`` units.
    keep : numpy.ndarray
        ``(N,)`` boolean inlier mask. After convergence it is the set ``sim`` was fitted
        to; if ``n_iter`` runs out, it is the set selected by the last pass.
    """
    keep = np.ones(len(src), bool)
    for _ in range(n_iter):
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        sigma = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = res <= max(k_sigma * sigma + np.median(res[keep]), floor)
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    return Similarity(s, R, t), res, keep


def pointing_offsets(R_est: np.ndarray, R_ref: np.ndarray, camera: FramingCamera):
    """Boresight offset and twist of an estimated attitude relative to a reference one.

    Parameters
    ----------
    R_est, R_ref : numpy.ndarray
        ``(3, 3)`` rotations that map body-fixed vectors into the camera frame: the
        estimate (e.g. COLMAP) and the reference (e.g. the label).
    camera : FramingCamera
        Supplies ``fx`` and ``fy`` to express the offset in pixels.

    Returns
    -------
    du, dv : float
        Position of the estimated boresight in the reference image relative to the
        principal point (px, distortion ignored).
    twist_deg : float
        Angle of the estimated camera x axis about the boresight, seen in the reference
        camera frame (deg).
    """
    b = R_ref @ R_est.T @ np.array([0.0, 0.0, 1.0])  # estimated boresight in the reference frame
    du = camera.fx * b[0] / b[2]
    dv = camera.fy * b[1] / b[2]
    x = R_ref @ R_est.T @ np.array([1.0, 0.0, 0.0])
    twist = np.degrees(np.arctan2(x[1], x[0]))
    return du, dv, twist


def georeference(model: SparseModel, meta: pd.DataFrame, body: Body, camera: FramingCamera):
    """Tie a COLMAP model to the body-fixed frame and check it against the label attitudes.

    A robust similarity (:func:`robust_similarity`) is fitted between the COLMAP camera
    centres and the label spacecraft positions. Each aligned COLMAP rotation is then
    compared with the label attitude; a median disagreement above 5 deg flags a mirrored
    or depth-reversed model and logs a warning.

    Parameters
    ----------
    model : SparseModel
        COLMAP model; only images whose name appears in ``meta`` are used.
    meta : pandas.DataFrame
        Image metadata (:func:`~asteroid_colmap.metadata.build_metadata`).
    body : Body
        Target body (rotation model).
    camera : FramingCamera
        Camera model, for the boresight offsets in pixels.

    Returns
    -------
    sim : Similarity
        Model-to-body transform (km).
    per_image : pandas.DataFrame
        One row per aligned image: ``image``, ``image_id``, aligned COLMAP centre
        ``sfm_x/y/z_km``, label position ``label_x/y/z_km``, ``position_residual_km``,
        ``used_in_fit``, ``attitude_residual_deg``, ``boresight_du_px``,
        ``boresight_dv_px``, ``twist_deg``, and ``sequence``, ``utc`` and ``range_km``
        from ``meta``.
    summary : dict
        ``num_aligned_images``, ``num_used_in_fit``, ``scale_km_per_unit``,
        ``position_rms_km`` (inliers only), ``position_max_km``, ``attitude_median_deg``,
        ``attitude_max_deg``, ``boresight_median_px`` and ``mirrored_or_reversed``.

    Raises
    ------
    RuntimeError
        If fewer than 3 registered images have label geometry.
    """
    lookup = meta.set_index("image")
    images = model.images[model.images["name"].isin(lookup.index)].reset_index(drop=True)
    if len(images) < 3:
        raise RuntimeError("fewer than 3 registered images with label geometry")
    geo = [label_geometry(lookup.loc[n], body) for n in images["name"]]
    C_sfm = np.stack(images["center"].to_numpy())
    C_lab = np.stack([g["sc_bf"] for g in geo])
    sim, res, keep = robust_similarity(C_sfm, C_lab)

    rows = []
    for i, (name, R_cw) in enumerate(zip(images["name"], images["R"])):
        R_sfm = R_cw @ sim.R.T  # body-fixed -> camera according to COLMAP
        R_lab = geo[i]["R_cam_from_bf"]
        du, dv, twist = pointing_offsets(R_sfm, R_lab, camera)
        C = sim.apply(C_sfm[i])
        rows.append({
            "image": name,
            "image_id": int(images.loc[i, "image_id"]),
            "sfm_x_km": C[0], "sfm_y_km": C[1], "sfm_z_km": C[2],
            "label_x_km": C_lab[i, 0], "label_y_km": C_lab[i, 1], "label_z_km": C_lab[i, 2],
            "position_residual_km": res[i],
            "used_in_fit": bool(keep[i]),
            "attitude_residual_deg": rotation_angle_deg(R_sfm, R_lab),
            "boresight_du_px": du, "boresight_dv_px": dv, "twist_deg": twist,
        })
    per_image = pd.DataFrame(rows).merge(meta[["image", "sequence", "utc", "range_km"]], on="image")
    att = per_image["attitude_residual_deg"]
    summary = {
        "num_aligned_images": int(len(images)),
        "num_used_in_fit": int(keep.sum()),
        "scale_km_per_unit": sim.scale,
        "position_rms_km": float(np.sqrt(np.mean(res[keep] ** 2))),
        "position_max_km": float(res.max()),
        "attitude_median_deg": float(att.median()),
        "attitude_max_deg": float(att.max()),
        "boresight_median_px": float(np.hypot(per_image.boresight_du_px,
                                              per_image.boresight_dv_px).median()),
        "mirrored_or_reversed": bool(att.median() > 5.0),
    }
    log.info("alignment: scale %.4g km/unit, position RMS %.3f km, attitude median %.4f deg",
             sim.scale, summary["position_rms_km"], summary["attitude_median_deg"])
    if summary["mirrored_or_reversed"]:
        log.warning("COLMAP attitudes disagree with the labels by %.1f deg (median): the model "
                    "is probably mirrored / depth-reversed - check the orientation step",
                    att.median())
    return sim, per_image, summary
