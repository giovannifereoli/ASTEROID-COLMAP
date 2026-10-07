import numpy as np
import pandas as pd

from asteroid_colmap.pairs import view_angle_matrix, view_angle_pairs, write_pairs


def _meta(angles_deg):
    a = np.radians(angles_deg)
    return pd.DataFrame({
        "image": [f"img{i}.png" for i in range(len(a))],
        "sc_bf_x_km": 5000 * np.cos(a), "sc_bf_y_km": 5000 * np.sin(a), "sc_bf_z_km": 0.0,
    })


def test_view_angle_matrix():
    ang = view_angle_matrix(_meta([0, 30, 90]))
    np.testing.assert_allclose(ang, [[0, 30, 90], [30, 0, 60], [90, 60, 0]], atol=1e-4)


def test_view_angle_pairs():
    meta = _meta([0, 30, 90])
    assert view_angle_pairs(meta, 40) == [("img0.png", "img1.png")]
    assert len(view_angle_pairs(meta, 100)) == 3
    assert view_angle_pairs(meta, 100, min_angle_deg=50) == [("img0.png", "img2.png"),
                                                             ("img1.png", "img2.png")]


def test_write_pairs(tmp_path):
    out = tmp_path / "colmap" / "pairs.txt"
    write_pairs([("a.png", "b.png"), ("a.png", "c.png")], out)
    assert out.read_text() == "a.png b.png\na.png c.png\n"
