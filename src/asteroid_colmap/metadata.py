"""Per-image observation geometry from the PDS3 labels.

Label conventions (verified on the Vesta RC3 labels):

* ``QUATERNION`` is scalar-first and rotates J2000 into the DAWN_FC2 camera frame.
* ``SC_TARGET_POSITION_VECTOR`` is the spacecraft -> target vector in **J2000** km, even
  though ``COORDINATE_SYSTEM_NAME = VESTA_FIXED`` (its norm is TARGET_CENTER_DISTANCE and its
  direction matches the boresight RA/Dec).
* ``SUB_SPACECRAFT_LATITUDE/LONGITUDE`` are planetocentric, east-positive, in the Claudia
  double-prime frame.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Body
from .geometry import body_from_j2000, cart_to_latlonr, latlon_to_unit, quat_to_matrix
from .pds import read_label

_SCALARS = {
    "exposure_ms": "EXPOSURE_DURATION",
    "ra_deg": "RIGHT_ASCENSION",
    "dec_deg": "DECLINATION",
    "twist_deg": "TWIST_ANGLE",
    "range_km": "TARGET_CENTER_DISTANCE",
    "altitude_km": "SPACECRAFT_ALTITUDE",
    "subsc_lat_deg": "SUB_SPACECRAFT_LATITUDE",
    "subsc_lon_deg": "SUB_SPACECRAFT_LONGITUDE",
    "subsolar_lat_deg": "SUB_SOLAR_LATITUDE",
    "subsolar_lon_deg": "SUB_SOLAR_LONGITUDE",
    "phase_deg": "PHASE_ANGLE",
    "incidence_deg": "INCIDENCE_ANGLE",
    "emission_deg": "EMISSION_ANGLE",
    "pixel_scale_km": "HORIZONTAL_PIXEL_SCALE",
    "center_lat_deg": "CENTER_LATITUDE",
    "center_lon_deg": "CENTER_LONGITUDE",
    "north_azimuth_deg": "NORTH_AZIMUTH",
}


def parse_time(label: dict) -> dt.datetime:
    """Image start time (UTC) from a parsed label.

    Uses ``DAWN:ALT_START_TIME`` (ISO calendar format) when present, otherwise
    ``START_TIME`` (day-of-year format ``YYYY-DDDTHH:MM:SS.fff``).

    Parameters
    ----------
    label : dict
        Output of :func:`~asteroid_colmap.pds.parse_label`.

    Returns
    -------
    datetime.datetime
        Naive UTC epoch.
    """
    alt = label.get("DAWN:ALT_START_TIME")
    if alt:
        return dt.datetime.fromisoformat(str(alt).rstrip("Z"))
    return dt.datetime.strptime(str(label["START_TIME"]).rstrip("Z"), "%Y-%jT%H:%M:%S.%f")


def label_record(lbl_path: Path, body: Body) -> dict:
    """One metadata row from a PDS3 label.

    Parameters
    ----------
    lbl_path : pathlib.Path
        Detached label; the image is expected next to it with the ``.FIT`` suffix.
    body : Body
        Target body, used to express the spacecraft position in the body-fixed frame.

    Returns
    -------
    dict
        ``image`` (PNG name), ``fit_path``, ``lbl_path``, ``sequence`` (from the
        directory name), ``observation_id``, ``filter``, ``utc`` (ISO, ms); the label
        scalars of :data:`_SCALARS` (``None`` when missing); the camera quaternion
        ``qw..qz`` (J2000 -> camera); ``sc2tgt_*_km`` and ``sc2sun_*_km`` (J2000);
        ``sc_bf_*_km`` (spacecraft position, body-fixed); ``rotation_model_check_deg``
        (angle between the label sub-spacecraft point and ``sc_bf``, only when the label
        gives the sub-spacecraft point); and ``subsc_lat_model_deg`` /
        ``subsc_lon_model_deg`` computed from ``sc_bf``.
    """
    lab = read_label(lbl_path)
    utc = parse_time(lab)
    fit = Path(lbl_path).with_suffix(".FIT")
    rec = {
        "image": fit.stem + ".png",
        "fit_path": str(fit),
        "lbl_path": str(lbl_path),
        "sequence": Path(lbl_path).parent.name.split("_", 1)[-1],
        "observation_id": lab.get("OBSERVATION_ID"),
        "filter": int(lab.get("FILTER_NUMBER", 0)),
        "utc": utc.isoformat(timespec="milliseconds"),
    }
    rec.update({k: lab.get(v) for k, v in _SCALARS.items()})
    rec.update(dict(zip(("qw", "qx", "qy", "qz"), lab["QUATERNION"])))
    tgt = np.array(lab["SC_TARGET_POSITION_VECTOR"], float)
    sun = np.array(lab["SC_SUN_POSITION_VECTOR"], float)
    rec.update(dict(zip(("sc2tgt_x_km", "sc2tgt_y_km", "sc2tgt_z_km"), tgt)))
    rec.update(dict(zip(("sc2sun_x_km", "sc2sun_y_km", "sc2sun_z_km"), sun)))

    # spacecraft position in the body-fixed frame, two independent ways
    sc_bf = body_from_j2000(body, utc) @ (-tgt)
    rec.update(dict(zip(("sc_bf_x_km", "sc_bf_y_km", "sc_bf_z_km"), sc_bf)))
    if rec["subsc_lat_deg"] is not None and rec["subsc_lon_deg"] is not None:
        u_lab = latlon_to_unit(rec["subsc_lat_deg"], rec["subsc_lon_deg"])
        rec["rotation_model_check_deg"] = float(
            np.degrees(np.arccos(np.clip(u_lab @ sc_bf / np.linalg.norm(sc_bf), -1, 1)))
        )
    lat, lon, _ = cart_to_latlonr(sc_bf)
    rec["subsc_lat_model_deg"], rec["subsc_lon_model_deg"] = float(lat[0]), float(lon[0])
    return rec


def build_metadata(raw_dir: Path, body: Body, out_csv: Path | None = None) -> pd.DataFrame:
    """Metadata table of every downloaded image, in time order.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Download root; searched recursively for ``*.LBL``.
    body : Body
        Target body.
    out_csv : pathlib.Path, optional
        If given, the table is also written there.

    Returns
    -------
    pandas.DataFrame
        One :func:`label_record` per label whose ``.FIT`` file exists, sorted by ``utc``.

    Raises
    ------
    FileNotFoundError
        If ``raw_dir`` holds no label.
    """
    labels = sorted(Path(raw_dir).rglob("*.LBL"))
    if not labels:
        raise FileNotFoundError(f"no .LBL files under {raw_dir}; run the download step first")
    df = pd.DataFrame([label_record(p, body) for p in labels])
    df = df[[Path(p).exists() for p in df.fit_path]]
    df = df.sort_values("utc").reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def load_metadata(path: Path) -> pd.DataFrame:
    """Read ``metadata.csv`` and add a parsed ``time`` column.

    Parameters
    ----------
    path : pathlib.Path
        File written by :func:`build_metadata`.

    Returns
    -------
    pandas.DataFrame
        The table, with ``time`` as pandas timestamps of ``utc``.
    """
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["utc"])
    return df


def label_geometry(row, body: Body) -> dict[str, np.ndarray]:
    """Body-fixed geometry of one metadata row.

    Parameters
    ----------
    row : pandas.Series or dict
        Metadata row with ``utc``, ``qw..qz``, ``sc2tgt_*_km`` and ``sc2sun_*_km``.
    body : Body
        Target body (rotation model).

    Returns
    -------
    dict of numpy.ndarray
        ``R_cam_from_bf``: ``(3, 3)`` label attitude, body-fixed -> camera.
        ``sc_bf``: ``(3,)`` spacecraft position (km).
        ``sun_bf``: ``(3,)`` unit vector from the body centre to the Sun.
    """
    utc = dt.datetime.fromisoformat(str(row["utc"]))
    R_bf = body_from_j2000(body, utc)
    R_cam = quat_to_matrix([row["qw"], row["qx"], row["qy"], row["qz"]])
    tgt = np.array([row["sc2tgt_x_km"], row["sc2tgt_y_km"], row["sc2tgt_z_km"]])
    sun = np.array([row["sc2sun_x_km"], row["sc2sun_y_km"], row["sc2sun_z_km"]]) - tgt
    return {
        "R_cam_from_bf": R_cam @ R_bf.T,
        "sc_bf": R_bf @ (-tgt),
        "sun_bf": R_bf @ (sun / np.linalg.norm(sun)),
    }
