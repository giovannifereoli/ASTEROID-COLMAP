"""Shared fixtures: a trimmed copy of a real Dawn FC2 label and the package presets."""

from __future__ import annotations

import pytest

from asteroid_colmap.camera import get_camera
from asteroid_colmap.config import VESTA

# FC21B0003112_11205060002F1C.LBL (first RC3 frame), keywords the pipeline reads plus a few
# that exercise the parser: units, "N/A", a value starting on the next line, nested OBJECT
# and a HISTORY object after END that must be ignored.
LABEL_TEXT = """\
PDS_VERSION_ID                = PDS3
LABEL_REVISION_NOTE           = "20080201, PGM, DAWN FC V1.5"

/* FILE CHARACTERISTICS */
INSTRUMENT_ID                 = "FC2"
OBSERVATION_ID                = "FC2_VSA_RC3"
OBSERVATION_TYPE              = "N/A"
START_TIME                    = 2011-205T06:00:02.137
DAWN:ALT_START_TIME           = 2011-07-24T06:00:02.137
FILTER_NUMBER                 = "1"
EXPOSURE_DURATION             = 8.000 <millisecond>
RIGHT_ASCENSION               = 295.784 <degree>
DECLINATION                   = -62.163 <degree>
TWIST_ANGLE                   = 247.186 <degree>
QUATERNION                    = (
    0.1744357085
    ,-0.3431161973
    ,-0.9079719753
    ,0.1656211065
)
SPICE_FILE_NAME               = (
    "sclk\\DAWN_203_SCLKSCET.00033.tsc"
    ,"lsk\\naif0010.tls"
)
COORDINATE_SYSTEM_NAME        = "VESTA_FIXED"
DESCRIPTION                   =
    "Geometry in this label is provided in the 'Claudia Double-Prime'
    coordinate system."
SUB_SPACECRAFT_LATITUDE       = 15.1976317794 <degree>
SUB_SPACECRAFT_LONGITUDE      = -58.3439166481 <degree>
SPACECRAFT_ALTITUDE           = 5202.287 <kilometer>
TARGET_CENTER_DISTANCE        = 5479.761 <kilometer>
SC_TARGET_POSITION_VECTOR     = (
    1102.517 <kilometer>
    ,-2285.015 <kilometer>
    ,-4857.051 <kilometer>
)
SUB_SOLAR_LATITUDE            = -26.7397100584 <degree>
SUB_SOLAR_LONGITUDE           = -48.7807607618 <degree>
SC_SUN_POSITION_VECTOR        = (
    -209014807.689 <kilometer>
    ,231091274.372 <kilometer>
    ,119358932.335 <kilometer>
)
PHASE_ANGLE                   = 42.7132783406 <degree>

OBJECT                        = IMAGE
    LINE_SAMPLES              = 1024
    LINES                     = 1024
    SAMPLE_TYPE               = "PC_REAL"
END_OBJECT                    = IMAGE

END
            OBJECT                        = HISTORY
GROUP                         = LEVEL_1A_GENERATION
    PHASE_ANGLE               = 999
END_GROUP                     = LEVEL_1A_GENERATION
END_OBJECT                    = HISTORY
END
"""


@pytest.fixture
def label_text() -> str:
    return LABEL_TEXT


@pytest.fixture
def label_path(tmp_path):
    d = tmp_path / "2011205_RC3"
    d.mkdir()
    p = d / "FC21B0003112_11205060002F1C.LBL"
    p.write_text(LABEL_TEXT, encoding="latin-1")
    return p


@pytest.fixture
def camera():
    return get_camera("dawn_fc2_f1")


@pytest.fixture
def body():
    return VESTA
