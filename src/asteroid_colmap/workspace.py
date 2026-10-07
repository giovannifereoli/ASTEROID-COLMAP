"""On-disk layout of a pipeline working directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self):
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=_jsonable))

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")
