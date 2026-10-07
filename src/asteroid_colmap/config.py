"""Dataset and target-body presets.

Adding another Dawn phase (e.g. Vesta Survey, or Ceres RC3 from DWNCFC2_1B) only needs a
new :class:`Dataset` entry: the archive layout, label keywords and camera are the same.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    """IAU-style rotation model and reference ellipsoid of a target body.

    The pole is fixed at ``(pole_ra_deg, pole_dec_deg)`` in J2000 and the prime meridian
    angle is ``W = w0_deg + w_rate_deg_per_day * d``, with ``d`` the TDB days since J2000
    (no precession or nutation terms).

    Attributes
    ----------
    name : str
        Body name.
    pole_ra_deg, pole_dec_deg : float
        Right ascension and declination of the north pole, J2000 (deg).
    w0_deg : float
        Prime meridian angle at J2000 (deg).
    w_rate_deg_per_day : float
        Rotation rate of the prime meridian (deg/day).
    radii_km : tuple of float
        Reference ellipsoid semi-axes ``(a, b, c)`` (km).
    frame_name : str
        Description of the body-fixed frame and its longitude convention.
    """

    name: str
    pole_ra_deg: float
    pole_dec_deg: float
    w0_deg: float
    w_rate_deg_per_day: float
    radii_km: tuple[float, float, float]
    frame_name: str


# IAU WGCCRE 2015 (Archinal et al. 2018) model, which defines the "Claudia double-prime"
# frame used by the Dawn FC PDS labels.  Checked against the labels: with W0 = 285.39 the
# label SC_TARGET_POSITION_VECTOR (J2000) rotates onto SUB_SPACECRAFT_LATITUDE/LONGITUDE to
# < 0.001 deg.  (The older dawn_vesta_v04.tpc "Claudia" W0 = 75.39 is off by 210 deg.)
# Ellipsoid: Russell et al. (2012), Science 336, 684.
VESTA = Body(
    name="Vesta",
    pole_ra_deg=309.031,
    pole_dec_deg=42.235,
    w0_deg=285.39,
    w_rate_deg_per_day=1617.3329428,
    radii_km=(286.3, 278.6, 223.2),
    frame_name="Claudia double-prime (IAU 2015), planetocentric, east-positive longitude",
)


@dataclass(frozen=True)
class Dataset:
    """A downloadable image set: archive location, target body and camera.

    Attributes
    ----------
    key : str
        Identifier used on the command line (``--dataset``).
    description : str
        One-line description.
    base_url : str
        Archive URL of the directory that holds ``subdirs``; ends with ``/``.
    subdirs : tuple of str
        Sub-directories (observation sequences) to list and download.
    body : Body
        Target body.
    camera : str
        Key in :data:`asteroid_colmap.camera.CAMERAS`.
    default_filters : tuple of int
        FC filter numbers downloaded when none are requested.
    """
    key: str
    description: str
    base_url: str
    subdirs: tuple[str, ...]
    body: Body
    camera: str = "dawn_fc2_f1"
    default_filters: tuple[int, ...] = (1,)


_VESTA_FC2_1B = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1B/DATA/FITS/"

DATASETS: dict[str, Dataset] = {
    "vesta-rc3": Dataset(
        key="vesta-rc3",
        description=(
            "Dawn FC2 calibrated images, Vesta approach, Rotational Characterization 3 "
            "(RC3 + RC3B), 2011-07-24/25, ~5200 km altitude, ~0.49 km/px"
        ),
        base_url=_VESTA_FC2_1B + "2011123_APPROACH/",
        subdirs=("2011205_RC3", "2011205_RC3B"),
        body=VESTA,
    ),
}


def get_dataset(key: str) -> Dataset:
    """Look up a dataset preset.

    Parameters
    ----------
    key : str
        Key in :data:`DATASETS`, e.g. ``"vesta-rc3"``.

    Returns
    -------
    Dataset

    Raises
    ------
    KeyError
        If ``key`` is unknown; the message lists the available keys.
    """
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; available: {', '.join(DATASETS)}") from None
