import numpy as np

from asteroid_colmap.model_io import count_registered_images, read_model

C45 = np.sqrt(0.5)

CAMERAS = """# Camera list with one line of data per camera:
1 OPENCV 1024 1024 10716.2 10723.1 512 512 0.1892 0 0 0
"""

IMAGES = f"""# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1 0 0 0 0 0 5 1 a.png
10.0 20.0 1 30.0 40.0 -1
2 {C45} 0 0 {C45} 1 2 3 1 b.png
50.0 60.0 1
3 1 0 0 0 0 0 0 1 c.png

"""

POINTS = """# 3D point list with one line of data per point:
1 0.5 0.5 10 120 130 140 0.3 1 0 2 0
"""


def test_read_model(tmp_path):
    (tmp_path / "cameras.txt").write_text(CAMERAS)
    (tmp_path / "images.txt").write_text(IMAGES)
    (tmp_path / "points3D.txt").write_text(POINTS)
    m = read_model(tmp_path)

    assert m.cameras.loc[0, "model"] == "OPENCV"
    assert m.cameras.loc[0, "params"][4] == 0.1892
    assert m.num_images == 3 and count_registered_images(tmp_path) == 3
    assert m.images["name"].tolist() == ["a.png", "b.png", "c.png"]

    # camera centre C = -R^T t
    np.testing.assert_allclose(m.images.loc[0, "center"], [0, 0, -5])
    R2 = m.images.loc[1, "R"]
    np.testing.assert_allclose(R2, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(m.images.loc[1, "center"], -R2.T @ [1, 2, 3])

    p = m.points.iloc[0]
    assert (p.point3d_id, p.track_length, p.error) == (1, 2, 0.3)
    obs = m.observations.sort_values("image_id")
    assert obs.image_id.tolist() == [1, 2]
    np.testing.assert_allclose(obs[["u", "v"]].to_numpy(), [[10, 20], [50, 60]])


def test_read_model_without_points(tmp_path):
    (tmp_path / "cameras.txt").write_text(CAMERAS)
    (tmp_path / "images.txt").write_text(IMAGES)
    (tmp_path / "points3D.txt").write_text("")
    m = read_model(tmp_path)
    assert len(m.points) == 0 and len(m.observations) == 0
    assert {"u", "v"} <= set(m.observations.columns)
