"""Reader for COLMAP sparse models in TXT format (``colmap model_converter --output_type TXT``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import quat_to_matrix


@dataclass
class SparseModel:
    """A COLMAP sparse model as tables.

    Attributes
    ----------
    cameras : pandas.DataFrame
        ``camera_id``, ``model``, ``width``, ``height``, ``params`` (list of float).
    images : pandas.DataFrame
        ``image_id``, ``name``, ``camera_id``, quaternion ``qw..qz`` and translation
        ``tx..tz`` (world -> camera), ``R`` (``(3, 3)`` rotation, world -> camera) and
        ``center`` (``(3,)`` camera centre in world coordinates).
    points : pandas.DataFrame
        ``point3d_id``, ``x``, ``y``, ``z``, colour ``r``, ``g``, ``b``, mean reprojection
        ``error`` (px) and ``track_length``.
    observations : pandas.DataFrame
        One row per track element: ``point3d_id``, ``image_id``, ``point2d_idx`` and the
        keypoint position ``u``, ``v`` (px, COLMAP convention).
    """
    cameras: pd.DataFrame  # camera_id, model, width, height, params
    images: pd.DataFrame  # image_id, name, camera_id, qw..qz, tx..tz, R (3x3), center (3,)
    points: pd.DataFrame  # point3d_id, x, y, z, r, g, b, error, track_length
    observations: pd.DataFrame  # point3d_id, image_id, point2d_idx, u, v

    @property
    def num_images(self) -> int:
        """Number of registered images."""
        return len(self.images)

    def image_centers(self) -> np.ndarray:
        """Camera centres in world coordinates.

        Returns
        -------
        numpy.ndarray
            ``(N, 3)``, in the row order of :attr:`images`.
        """
        return np.stack(self.images["center"].to_numpy())

    def image_rotations(self) -> np.ndarray:
        """World -> camera rotations.

        Returns
        -------
        numpy.ndarray
            ``(N, 3, 3)``, in the row order of :attr:`images`.
        """
        return np.stack(self.images["R"].to_numpy())


def _data_lines(path: Path):
    """Yield the non-empty, non-comment lines of a COLMAP TXT file."""
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                yield line.rstrip("\n")


def read_cameras(path: Path) -> pd.DataFrame:
    """Read ``cameras.txt``.

    Parameters
    ----------
    path : pathlib.Path
        COLMAP ``cameras.txt``.

    Returns
    -------
    pandas.DataFrame
        ``camera_id``, ``model``, ``width``, ``height`` and ``params`` (list of float).
    """
    rows = []
    for line in _data_lines(path):
        p = line.split()
        rows.append({"camera_id": int(p[0]), "model": p[1], "width": int(p[2]),
                     "height": int(p[3]), "params": [float(v) for v in p[4:]]})
    return pd.DataFrame(rows)


def read_images(path: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Read ``images.txt``: image poses and their keypoints.

    Parameters
    ----------
    path : pathlib.Path
        COLMAP ``images.txt``.

    Returns
    -------
    images : pandas.DataFrame
        See :attr:`SparseModel.images`; ``center = -R^T t``.
    keypoints : dict of int to numpy.ndarray
        Per ``image_id``, an ``(N, 3)`` array of ``x``, ``y`` (px) and ``point3d_id``
        (-1 when the keypoint has no 3-D point).
    """
    rows, kps = [], {}
    lines = _data_lines_keep_empty(path)
    for header, pts in zip(lines[0::2], lines[1::2]):
        p = header.split()
        image_id = int(p[0])
        q = np.array(p[1:5], float)
        t = np.array(p[5:8], float)
        R = quat_to_matrix(q)
        rows.append({"image_id": image_id, "name": p[9], "camera_id": int(p[8]),
                     "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                     "tx": t[0], "ty": t[1], "tz": t[2], "R": R, "center": -R.T @ t})
        arr = np.array(pts.split(), float) if pts.strip() else np.empty(0)
        kps[image_id] = arr.reshape(-1, 3)
    return pd.DataFrame(rows), kps


def _data_lines_keep_empty(path: Path) -> list[str]:
    """Non-comment lines of a COLMAP TXT file, empty lines included.

    In ``images.txt`` every image takes exactly two lines and the keypoint line may be
    empty, so empty lines must be kept to stay in step.
    """
    with open(path) as fh:
        return [ln.rstrip("\n") for ln in fh if not ln.startswith("#")]


def read_points(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``points3D.txt``: 3-D points and their tracks.

    Parameters
    ----------
    path : pathlib.Path
        COLMAP ``points3D.txt``.

    Returns
    -------
    points : pandas.DataFrame
        ``point3d_id``, ``x``, ``y``, ``z``, ``r``, ``g``, ``b``, ``error`` (px) and
        ``track_length``.
    observations : pandas.DataFrame
        One row per track element: ``point3d_id``, ``image_id``, ``point2d_idx``.
    """
    pts, tracks = [], []
    for line in _data_lines(path):
        p = line.split()
        pid = int(p[0])
        track = np.array(p[8:], int).reshape(-1, 2)
        pts.append((pid, *map(float, p[1:4]), *map(int, p[4:7]), float(p[7]), len(track)))
        tracks.append(np.column_stack([np.full(len(track), pid), track]))
    points = pd.DataFrame(pts, columns=["point3d_id", "x", "y", "z", "r", "g", "b", "error",
                                        "track_length"])
    obs = np.concatenate(tracks) if tracks else np.empty((0, 3), int)
    return points, pd.DataFrame(obs, columns=["point3d_id", "image_id", "point2d_idx"])


def read_model(model_dir: str | Path) -> SparseModel:
    """Read a COLMAP TXT model directory.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Directory with ``cameras.txt``, ``images.txt`` and ``points3D.txt``.

    Returns
    -------
    SparseModel
        With the keypoint position ``u``, ``v`` of every observation looked up from
        ``images.txt``.
    """
    d = Path(model_dir)
    cameras = read_cameras(d / "cameras.txt")
    images, kps = read_images(d / "images.txt")
    points, obs = read_points(d / "points3D.txt")
    if len(obs):
        uv = np.empty((len(obs), 2))
        for image_id, idx in obs.groupby("image_id").groups.items():
            uv[idx] = kps[image_id][obs.loc[idx, "point2d_idx"].to_numpy(), :2]
        obs["u"], obs["v"] = uv[:, 0], uv[:, 1]
    else:
        obs["u"], obs["v"] = [], []
    return SparseModel(cameras, images, points, obs)


def count_registered_images(model_dir: str | Path) -> int:
    """Number of images in a COLMAP TXT model, without parsing it.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Directory with ``images.txt``.

    Returns
    -------
    int
    """
    return len(_data_lines_keep_empty(Path(model_dir) / "images.txt")) // 2
