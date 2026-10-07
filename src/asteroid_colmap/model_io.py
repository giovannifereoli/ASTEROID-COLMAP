"""Reader for COLMAP sparse models in TXT format (``colmap model_converter --output_type TXT``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import quat_to_matrix


@dataclass
class SparseModel:
    cameras: pd.DataFrame  # camera_id, model, width, height, params
    images: pd.DataFrame  # image_id, name, camera_id, qw..qz, tx..tz, R (3x3), center (3,)
    points: pd.DataFrame  # point3d_id, x, y, z, r, g, b, error, track_length
    observations: pd.DataFrame  # point3d_id, image_id, point2d_idx, u, v

    @property
    def num_images(self) -> int:
        return len(self.images)

    def image_centers(self) -> np.ndarray:
        return np.stack(self.images["center"].to_numpy())

    def image_rotations(self) -> np.ndarray:
        return np.stack(self.images["R"].to_numpy())


def _data_lines(path: Path):
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                yield line.rstrip("\n")


def read_cameras(path: Path) -> pd.DataFrame:
    rows = []
    for line in _data_lines(path):
        p = line.split()
        rows.append({"camera_id": int(p[0]), "model": p[1], "width": int(p[2]),
                     "height": int(p[3]), "params": [float(v) for v in p[4:]]})
    return pd.DataFrame(rows)


def read_images(path: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Image poses and, per image, the (N, 3) array of keypoints ``x, y, point3d_id``."""
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
    # every image takes exactly two lines; the keypoint line may be empty
    with open(path) as fh:
        return [ln.rstrip("\n") for ln in fh if not ln.startswith("#")]


def read_points(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return len(_data_lines_keep_empty(Path(model_dir) / "images.txt")) // 2
