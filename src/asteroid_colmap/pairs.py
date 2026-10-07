"""Geometry-aware image pair selection.

Images are paired when the spacecraft directions seen from the body centre (body-fixed
frame, from the labels) differ by less than ``max_angle_deg``. With the Sun fixed in inertial
space this also bounds the illumination change, and it pairs images from different
sequences (RC3 / RC3B) that look at the same hemisphere - which a purely sequential matcher
would miss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def view_directions(meta: pd.DataFrame) -> np.ndarray:
    """Unit vectors from the body centre to the spacecraft, body-fixed frame.

    Parameters
    ----------
    meta : pandas.DataFrame
        Metadata with ``sc_bf_x_km``, ``sc_bf_y_km`` and ``sc_bf_z_km``.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` unit vectors in row order.
    """
    p = meta[["sc_bf_x_km", "sc_bf_y_km", "sc_bf_z_km"]].to_numpy(float)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def view_angle_matrix(meta: pd.DataFrame) -> np.ndarray:
    """Angle between the spacecraft directions of every pair of images.

    Parameters
    ----------
    meta : pandas.DataFrame
        See :func:`view_directions`.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(N, N)`` matrix of angles (deg), zero on the diagonal.
    """
    u = view_directions(meta)
    return np.degrees(np.arccos(np.clip(u @ u.T, -1.0, 1.0)))


def view_angle_pairs(
    meta: pd.DataFrame, max_angle_deg: float = 40.0, min_angle_deg: float = 0.0
) -> list[tuple[str, str]]:
    """Image pairs whose spacecraft directions differ by an angle within the given bounds.

    Parameters
    ----------
    meta : pandas.DataFrame
        Metadata with ``image`` and the ``sc_bf_*_km`` columns.
    max_angle_deg, min_angle_deg : float
        Inclusive bounds on the angle between the two directions (deg).

    Returns
    -------
    list of tuple of str
        ``(name_i, name_j)`` with ``i < j`` in row order; each pair appears once.
    """
    ang = view_angle_matrix(meta)
    i, j = np.where(np.triu((ang <= max_angle_deg) & (ang >= min_angle_deg), k=1))
    names = meta["image"].to_numpy()
    return [(names[a], names[b]) for a, b in zip(i, j)]


def write_pairs(pairs: list[tuple[str, str]], path: Path) -> None:
    """Write image pairs for COLMAP ``matches_importer``, one ``name_a name_b`` per line.

    Parameters
    ----------
    pairs : list of tuple of str
        Image-name pairs.
    path : pathlib.Path
        Output file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
