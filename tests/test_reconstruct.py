import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from asteroid_colmap.camera import FramingCamera
from asteroid_colmap.metadata import label_geometry, label_record
from asteroid_colmap.model_io import read_model
from asteroid_colmap.reconstruct import _camera_limits, write_label_model
from asteroid_colmap.workspace import Workspace


def test_camera_limits_relax_focal_ratio_for_narrow_angle_camera(camera):
    """fx / width = 10.47 for the FC: above COLMAP's default limit of 10, which would make
    the mapper discard the camera as bogus and triangulate nothing."""
    limit = _camera_limits(camera)["Mapper.max_focal_length_ratio"]
    assert camera.fx / camera.width > 10.0
    assert limit == pytest.approx(2 * camera.fy / 1024)
    assert limit == pytest.approx(20.94, abs=0.01)


def test_camera_limits_keep_default_for_ordinary_camera():
    wide = FramingCamera("wide", 20.0, (0.01, 0.01), (511.5, 511.5), 0.0, 1024, 1024)
    assert _camera_limits(wide) == {"Mapper.max_focal_length_ratio": 10.0}


def _database(path, rows):
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as con:
        con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
        con.executemany("INSERT INTO images VALUES (?, ?, ?)", rows)
        con.commit()


def test_write_label_model_round_trip(tmp_path, label_path, body, camera):
    rec = label_record(label_path, body)
    meta = pd.DataFrame([rec])
    ws = Workspace(tmp_path / "work")
    # the database also holds an image without label geometry, which must be skipped
    _database(ws.database, [(1, rec["image"], 1), (2, "unknown.png", 1)])

    out = tmp_path / "label_poses"
    assert write_label_model(ws, meta, body, camera, out) == 1
    lines = (out / "images.txt").read_text().split("\n")
    assert lines[0].split()[-2:] == ["1", rec["image"]] and lines[1] == ""
    assert (out / "points3D.txt").read_text() == ""

    m = read_model(out)
    assert m.cameras.loc[0, "model"] == "OPENCV"
    np.testing.assert_allclose(m.cameras.loc[0, "params"], camera.colmap_params)
    g = label_geometry(rec, body)
    np.testing.assert_allclose(m.images.loc[0, "R"], g["R_cam_from_bf"], atol=1e-12)
    np.testing.assert_allclose(m.images.loc[0, "center"], g["sc_bf"], atol=1e-8)
