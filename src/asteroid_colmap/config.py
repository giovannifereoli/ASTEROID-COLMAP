"""Dataset and target-body presets.

Adding another Dawn phase (e.g. Vesta Survey, or Ceres RC3 from DWNCFC2_1B) only needs a
new :class:`Dataset` entry: the archive layout, label keywords and camera are the same.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    """IAU-style rotation model and reference ellipsoid of a target body."""

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
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; available: {', '.join(DATASETS)}") from None
