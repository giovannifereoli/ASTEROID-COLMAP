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
    """Pinhole framing camera with one radial distortion term, in NAIF IK terms.

    Instances are immutable; the COLMAP ``OPENCV`` parameters are derived from the IK
    constants (see the module docstring for the mapping).

    Attributes
    ----------
    name : str
        Human-readable camera and filter name.
    focal_length_mm : float
        Effective focal length FL (mm).
    pixel_pitch_mm : tuple of float
        Detector pixel pitch ``(x, y)`` (mm/px), i.e. ``1 / Kx`` and ``1 / Ky``.
    center_px : tuple of float
        Optical centre ``(S0, L0)`` in IK convention: 0-based pixel centres (px).
    radial_e1_per_mm2 : float
        IK radial distortion coefficient E1 (1/mm^2).
    width, height : int
        Image size (px).
    """
    name: str
    focal_length_mm: float
    pixel_pitch_mm: tuple[float, float]
    center_px: tuple[float, float]  # IK convention (0-based pixel centres)
    radial_e1_per_mm2: float
    width: int
    height: int

    @property
    def fx(self) -> float:
        """Focal length along the image x axis (px): ``FL / pitch_x``."""
        return self.focal_length_mm / self.pixel_pitch_mm[0]

    @property
    def fy(self) -> float:
        """Focal length along the image y axis (px): ``FL / pitch_y``."""
        return self.focal_length_mm / self.pixel_pitch_mm[1]

    @property
    def cx(self) -> float:
        """Principal point x in COLMAP pixel coordinates (px): IK centre + 0.5."""
        return self.center_px[0] + 0.5

    @property
    def cy(self) -> float:
        """Principal point y in COLMAP pixel coordinates (px): IK centre + 0.5."""
        return self.center_px[1] + 0.5

    @property
    def k1(self) -> float:
        """COLMAP radial coefficient for normalised coordinates: ``E1 * FL**2``."""
        return self.radial_e1_per_mm2 * self.focal_length_mm**2

    @property
    def colmap_model(self) -> str:
        """COLMAP camera model name, always ``"OPENCV"``."""
        return "OPENCV"

    @property
    def colmap_params(self) -> list[float]:
        """COLMAP ``OPENCV`` parameters ``[fx, fy, cx, cy, k1, k2, p1, p2]``; k2 = p1 = p2 = 0."""
        return [self.fx, self.fy, self.cx, self.cy, self.k1, 0.0, 0.0, 0.0]

    def colmap_params_str(self) -> str:
        """COLMAP parameters as a comma-separated string, for ``ImageReader.camera_params``.

        Returns
        -------
        str
            :attr:`colmap_params` formatted with ``.10g``.
        """
        return ",".join(f"{v:.10g}" for v in self.colmap_params)

    @property
    def ifov_rad(self) -> float:
        """Instantaneous field of view of one pixel along x (rad): ``1 / fx``."""
        return 1.0 / self.fx

    def project(self, X_cam: np.ndarray) -> np.ndarray:
        """Project camera-frame points to COLMAP pixel coordinates.

        Applies the pinhole projection, then the radial distortion ``1 + k1 (x^2 + y^2)`` on
        normalised coordinates. Points behind the camera are not rejected.

        Parameters
        ----------
        X_cam : numpy.ndarray
            ``(N, 3)`` or ``(3,)`` points in the camera frame (x right, y down, z along the
            boresight), any length unit.

        Returns
        -------
        numpy.ndarray
            ``(N, 2)`` pixel coordinates ``(u, v)``, pixel centres at +0.5 (COLMAP convention).
        """
        X_cam = np.atleast_2d(X_cam)
        x = X_cam[:, 0] / X_cam[:, 2]
        y = X_cam[:, 1] / X_cam[:, 2]
        d = 1.0 + self.k1 * (x * x + y * y)
        return np.column_stack([self.fx * x * d + self.cx, self.fy * y * d + self.cy])

    def pixel_rays(self, u: np.ndarray, v: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Back-project COLMAP pixel coordinates to unit camera-frame rays.

        The radial distortion is removed by fixed-point iteration; the correction is below
        0.1 % across the 5.5 deg field of view, so a few iterations are enough.

        Parameters
        ----------
        u, v : array_like
            Pixel coordinates (px, COLMAP convention), any matching shape.
        iterations : int
            Number of fixed-point iterations.

        Returns
        -------
        numpy.ndarray
            Unit rays of shape ``u.shape + (3,)`` in the camera frame.
        """
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
    """Look up a camera preset.

    Parameters
    ----------
    key : str
        Key in :data:`CAMERAS`, e.g. ``"dawn_fc2_f1"``.

    Returns
    -------
    FramingCamera

    Raises
    ------
    KeyError
        If ``key`` is unknown; the message lists the available keys.
    """
    try:
        return CAMERAS[key]
    except KeyError:
        raise KeyError(f"unknown camera {key!r}; available: {', '.join(CAMERAS)}") from None
