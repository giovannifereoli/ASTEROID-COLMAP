"""On-disk layout of a pipeline working directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workspace:
    """Paths of a pipeline working directory.

    Every pipeline step reads and writes under ``root``. The properties only build paths;
    nothing is created until a step writes there.

    Attributes
    ----------
    root : pathlib.Path
        Working directory, made absolute with ``~`` expanded.
    raw : pathlib.Path
        ``raw/``: untouched PDS downloads (FITS + LBL), one sub-directory per sequence.
    metadata_csv : pathlib.Path
        ``metadata.csv``: per-image label geometry.
    images : pathlib.Path
        ``images/``: 8-bit PNGs fed to COLMAP.
    masks : pathlib.Path
        ``masks/``: COLMAP feature masks.
    prepare_json : pathlib.Path
        ``prepare.json``: preprocessing settings, orientation scores and mask fractions.
    colmap : pathlib.Path
        ``colmap/``: COLMAP database, pairs, models and logs.
    database : pathlib.Path
        ``colmap/database.db``: COLMAP feature and match database.
    pairs : pathlib.Path
        ``colmap/pairs.txt``: image pairs for ``matches_importer``.
    sparse : pathlib.Path
        ``colmap/sparse``: binary sparse models.
    model_txt : pathlib.Path
        ``colmap/model_txt``: the same models in TXT format.
    reconstruct_json : pathlib.Path
        ``colmap/reconstruct.json``: COLMAP version, options, models and timings.
    logs : pathlib.Path
        ``colmap/logs``: one log file per COLMAP call.
    catalog : pathlib.Path
        ``catalog/``: landmark catalog tables, point cloud and ``summary.json``.
    plots : pathlib.Path
        ``plots/``: figures.
    """
    root: Path

    def __post_init__(self):
        """Normalise ``root`` to an absolute path with ``~`` expanded."""
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    raw = property(lambda self: self.root / "raw")  # untouched PDS downloads
    metadata_csv = property(lambda self: self.root / "metadata.csv")
    images = property(lambda self: self.root / "images")  # 8-bit PNGs fed to COLMAP
    masks = property(lambda self: self.root / "masks")  # COLMAP feature masks
    prepare_json = property(lambda self: self.root / "prepare.json")
    colmap = property(lambda self: self.root / "colmap")
    database = property(lambda self: self.root / "colmap" / "database.db")
    pairs = property(lambda self: self.root / "colmap" / "pairs.txt")
    sparse = property(lambda self: self.root / "colmap" / "sparse")
    model_txt = property(lambda self: self.root / "colmap" / "model_txt")
    reconstruct_json = property(lambda self: self.root / "colmap" / "reconstruct.json")
    logs = property(lambda self: self.root / "colmap" / "logs")
    catalog = property(lambda self: self.root / "catalog")
    plots = property(lambda self: self.root / "plots")

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write a dictionary as indented JSON, creating parent directories.

        NumPy scalars and arrays and :class:`pathlib.Path` values are converted on the fly.

        Parameters
        ----------
        path : pathlib.Path
            Output file.
        data : dict
            Mapping to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=_jsonable))

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        """Read a JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Input file.

        Returns
        -------
        dict
        """
        return json.loads(Path(path).read_text())


def _jsonable(obj):
    """``json.dumps`` fallback: NumPy scalars and arrays to Python values, paths to strings.

    Raises
    ------
    TypeError
        For any other type.
    """
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")
