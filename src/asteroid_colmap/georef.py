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
    scale: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(X) @ self.R.T + self.t

    def as_dict(self) -> dict:
        return {"scale_km_per_unit": self.scale, "R": self.R.tolist(), "t_km": self.t.tolist()}


def robust_similarity(src, dst, n_iter: int = 5, k_sigma: float = 4.0, floor: float = 1.0):
    """Umeyama fit with iterative rejection of residuals > max(k_sigma * 1.4826 MAD, floor)."""
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
    """Offset of the estimated boresight in the reference image (px) and twist (deg).

    ``R_*`` map body-fixed vectors into the camera frame.
    """
    b = R_ref @ R_est.T @ np.array([0.0, 0.0, 1.0])  # estimated boresight in the reference frame
    du = camera.fx * b[0] / b[2]
    dv = camera.fy * b[1] / b[2]
    x = R_ref @ R_est.T @ np.array([1.0, 0.0, 0.0])
    twist = np.degrees(np.arctan2(x[1], x[0]))
    return du, dv, twist


def georeference(model: SparseModel, meta: pd.DataFrame, body: Body, camera: FramingCamera):
    """Returns (similarity, per-image DataFrame of residuals, summary dict)."""
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
