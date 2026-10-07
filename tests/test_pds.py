from asteroid_colmap.pds import parse_label, parse_listing


def test_scalars_units_and_strings(label_text):
    lab = parse_label(label_text)
    assert lab["PDS_VERSION_ID"] == "PDS3"
    assert lab["OBSERVATION_ID"] == "FC2_VSA_RC3"
    assert lab["FILTER_NUMBER"] == "1"
    assert lab["EXPOSURE_DURATION"] == 8.0
    assert lab["TARGET_CENTER_DISTANCE"] == 5479.761
    assert lab["START_TIME"] == "2011-205T06:00:02.137"
    assert lab["OBSERVATION_TYPE"] is None


def test_multiline_vectors(label_text):
    lab = parse_label(label_text)
    assert lab["QUATERNION"] == (0.1744357085, -0.3431161973, -0.9079719753, 0.1656211065)
    assert lab["SC_TARGET_POSITION_VECTOR"] == (1102.517, -2285.015, -4857.051)
    assert lab["SPICE_FILE_NAME"][1] == "lsk\\naif0010.tls"


def test_value_on_next_line_is_joined(label_text):
    desc = parse_label(label_text)["DESCRIPTION"]
    assert desc.startswith("Geometry in this label")
    assert desc.endswith("coordinate system.")
    assert "  " not in desc


def test_objects_flattened_and_history_after_end_ignored(label_text):
    lab = parse_label(label_text)
    assert lab["IMAGE.LINES"] == 1024
    assert lab["IMAGE.SAMPLE_TYPE"] == "PC_REAL"
    assert lab["PHASE_ANGLE"] == 42.7132783406
    assert not any(k.startswith("HISTORY") for k in lab)


def test_parse_listing():
    html = """<a href="?C=N;O=D">Name</a> <a href="/pds3/dawn/">Parent</a>
              <a href="FC21B0003112_11205060002F1C.FIT">x</a>
              <a href="FC21B0003112_11205060002F1C.LBL">x</a>
              <a href="2011205_RC3/">dir</a>"""
    assert parse_listing(html) == ["FC21B0003112_11205060002F1C.FIT",
                                   "FC21B0003112_11205060002F1C.LBL"]
