"""Framing-camera intrinsics expressed as a COLMAP ``OPENCV`` camera.

Dawn FC model (NAIF IK ``dawn_fc_v10.ti``)::

    (X, Y) = FL * (P1, P2) / P3                    focal-plane mm
    (dX, dY) = E1 * R^2 * (X, Y)                   radial distortion, R^2 = X^2 + Y^2
    (S, L) = (Kx (X + dX), Ky (Y + dY)) + (S0, L0)  pixels, Kx = 1 / pixel pitch

With normalised coordinates x = X / FL this is exactly COLMAP's OPENCV model with
fx = FL Kx, fy = FL Ky and k1 = E1 FL^2 (k2 = p1 = p2 = 0). The IK centre (511.5, 511.5) is
in 0-based pixel-centre coordinates; COLMAP puts pixel centres at +0.5, hence cx = cy = 512.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramingCamera:
    name: str
    focal_length_mm: float
    pixel_pitch_mm: tuple[float, float]
    center_px: tuple[float, float]  # IK convention (0-based pixel centres)
    radial_e1_per_mm2: float
    width: int
    height: int

    @property
    def fx(self) -> float:
        return self.focal_length_mm / self.pixel_pitch_mm[0]

    @property
    def fy(self) -> float:
        return self.focal_length_mm / self.pixel_pitch_mm[1]

    @property
    def cx(self) -> float:
        return self.center_px[0] + 0.5

    @property
    def cy(self) -> float:
        return self.center_px[1] + 0.5

    @property
    def k1(self) -> float:
        return self.radial_e1_per_mm2 * self.focal_length_mm**2

    @property
    def colmap_model(self) -> str:
        return "OPENCV"

    @property
    def colmap_params(self) -> list[float]:
        return [self.fx, self.fy, self.cx, self.cy, self.k1, 0.0, 0.0, 0.0]

    def colmap_params_str(self) -> str:
        return ",".join(f"{v:.10g}" for v in self.colmap_params)

    @property
    def ifov_rad(self) -> float:
        return 1.0 / self.fx

    def project(self, X_cam: np.ndarray) -> np.ndarray:
        """Camera-frame points (N, 3) -> COLMAP pixel coordinates (N, 2)."""
        X_cam = np.atleast_2d(X_cam)
        x = X_cam[:, 0] / X_cam[:, 2]
        y = X_cam[:, 1] / X_cam[:, 2]
        d = 1.0 + self.k1 * (x * x + y * y)
        return np.column_stack([self.fx * x * d + self.cx, self.fy * y * d + self.cy])

    def pixel_rays(self, u: np.ndarray, v: np.ndarray, iterations: int = 3) -> np.ndarray:
        """COLMAP pixel coordinates -> unit camera-frame rays (..., 3), distortion removed."""
        xd = (np.asarray(u, float) - self.cx) / self.fx
        yd = (np.asarray(v, float) - self.cy) / self.fy
        x, y = xd, yd
        for _ in range(iterations):
            d = 1.0 + self.k1 * (x * x + y * y)
            x, y = xd / d, yd / d
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


CAMERAS: dict[str, FramingCamera] = {
    # FC2 (NAIF ID -203121), clear filter F1
    "dawn_fc2_f1": FramingCamera(
        name="Dawn FC2, filter 1 (clear)",
        focal_length_mm=150.07,
        pixel_pitch_mm=(0.014004, 0.013995),
        center_px=(511.5, 511.5),
        radial_e1_per_mm2=8.4e-6,
        width=1024,
        height=1024,
    ),
}


def get_camera(key: str) -> FramingCamera:
    try:
        return CAMERAS[key]
    except KeyError:
        raise KeyError(f"unknown camera {key!r}; available: {', '.join(CAMERAS)}") from None
