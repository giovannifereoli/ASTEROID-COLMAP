import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from asteroid_colmap.camera import FramingCamera
from asteroid_colmap.metadata import label_geometry, label_record
from asteroid_colmap.model_io import read_model
from asteroid_colmap import reconstruct
from asteroid_colmap.reconstruct import ReconstructionOptions, _camera_limits, write_label_model
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


def test_extract_features_uses_one_orientation_per_keypoint(tmp_path, monkeypatch, camera):
    """A second SIFT orientation is a second keypoint at the same pixel, which COLMAP can
    triangulate into a duplicate landmark; the default must keep one, and must also clean
    the database because the covariant (DSP) extractor ignores the option."""
    calls = []
    monkeypatch.setattr(reconstruct, "run_colmap", lambda exe, cmd, options, log_dir: calls.append((cmd, options)))
    monkeypatch.setattr(reconstruct, "drop_duplicate_keypoints", lambda db: calls.append(db) or (0, 0))
    ws = Workspace(tmp_path)
    reconstruct.extract_features("colmap", ws, camera, ReconstructionOptions())
    (cmd, options), db = calls
    assert cmd == "feature_extractor"
    assert options["SiftExtraction.max_num_orientations"] == 1
    assert db == ws.database


def test_drop_duplicate_keypoints_keeps_first_copy_per_pixel(tmp_path):
    """Keypoints and descriptors stay row-aligned; images without duplicates are untouched."""
    db = tmp_path / "database.db"
    kp1 = np.array([[10.5, 20.5, 1, 0, 0, 1],     # first orientation
                    [30.5, 40.5, 1, 0, 0, 1],
                    [10.5, 20.5, 0, 1, -1, 0],    # second orientation, same pixel
                    [50.5, 60.5, 1, 0, 0, 1]], np.float32)
    desc1 = np.arange(4 * 128, dtype=np.uint8).reshape(4, 128)
    kp2 = kp1[[0, 1, 3]]
    desc2 = desc1[[0, 1, 3]]
    with closing(sqlite3.connect(db)) as con, con:
        con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        con.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, type INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
        for i, (k, d) in enumerate([(kp1, desc1), (kp2, desc2)], start=1):
            con.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?)", (i, len(k), 6, k.tobytes()))
            con.execute("INSERT INTO descriptors VALUES (?, 0, ?, 128, ?)", (i, len(d), d.tobytes()))

    assert reconstruct.drop_duplicate_keypoints(db) == (7, 6)
    assert reconstruct.drop_duplicate_keypoints(db) == (6, 6)  # idempotent
    with closing(sqlite3.connect(db)) as con:
        rows = con.execute("SELECT k.rows, k.data, d.rows, d.data FROM keypoints k "
                           "JOIN descriptors d USING (image_id) ORDER BY image_id").fetchall()
    for (nk, kdata, nd, ddata), (k, d) in zip(rows, [(kp2, desc2), (kp2, desc2)]):
        assert nk == nd == 3
        np.testing.assert_array_equal(np.frombuffer(kdata, np.float32).reshape(3, 6), k)
        np.testing.assert_array_equal(np.frombuffer(ddata, np.uint8).reshape(3, 128), d)


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
