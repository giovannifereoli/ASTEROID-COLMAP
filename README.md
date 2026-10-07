# asteroid-colmap

Download the Dawn Framing Camera images of Vesta's RC3 approach from the PDS and turn them into a
georeferenced **landmark catalog** using [COLMAP](https://github.com/colmap/colmap). The package also
writes figures for every stage.

```
PDS FITS + labels ─► 8-bit PNG + body masks ─► COLMAP SIFT / matching / mapping
                 ─► similarity fit to the label poses ─► body-fixed landmarks (Claudia″) ─► plots
```

> **Hera or Dawn?** These images come from NASA's **Dawn** mission. ESA's Hera carries Framing
> Cameras built on the same Dawn FC heritage, but Hera never visited Vesta. RC3
> ("Rotational Characterization 3") is a Dawn Vesta approach sub-phase, flown on 24–25 July 2011.

<p align="center">
  <img src="examples/vesta_rc3/plots/08_landmark_map.png" width="900" alt="Landmark map of Vesta">
</p>

## The Vesta RC3 result

130 clear-filter (F1) images from FC2. Archive: `DWNVFC2_1B/DATA/FITS/2011123_APPROACH/`, sub-directories
`2011205_RC3` and `2011205_RC3B`. Each sequence covers one full Vesta rotation of 5.3 h from a range of
about 5,500 km, at 0.49 km/px. RC3 views latitudes 15° to 4° at a phase angle of 43° to 32°. RC3B views
latitudes −16° to −27° at a phase angle of 13° to 8°.

| | label-poses mode (default) | incremental mode (check) |
|---|---|---|
| registered images | 130 / 130 | 130 / 130 |
| landmarks (track ≥ 3) | 110,281 | 109,821 |
| grades A / B / C | 43,377 / 46,662 / 20,242 | 43,776 / 46,316 / 19,729 |
| curated set (one per 10 × 10 km cell) | 7,509 (A 6,625 / B 884) | 7,517 |
| observations | 1,044,759 | 1,050,617 |
| median track length / reprojection error | 7 / 0.24 px | 7 / 0.24 px |
| median triangulation angle | 34.7° | 35.0° |
| camera-centre residual vs label (RMS) | 2.26 km | 2.10 km |
| attitude difference vs label (median) | 79″ | 78″ |
| COLMAP mapping time (CPU) | **35 s** | 284 s |

The two modes match 98,975 landmarks to each other through shared keypoints. The two catalogs differ by
a common offset of about 0.12 km, and after removing it they agree to **15 m median and 45 m p90**.
See [Validation](#validation-two-independent-mapping-modes) below.

Heights above the 286.3 × 278.6 × 223.2 km ellipsoid range from −9.8 km (p5) to +8.2 km (p95).
99 % of the landmarks lie between 74°S and 44°N; the curated set reaches 89°S and 56°N. The north had
not been mapped yet because it was in polar night during the approach.

## Installation

```bash
pip install -e ".[interactive,dev]"     # Python >= 3.10; plotly for the 3-D HTML page, pytest
```

COLMAP is driven through its command-line interface, so you need the `colmap` binary:

```bash
brew install colmap                     # macOS
conda install -c conda-forge colmap     # any platform
sudo apt install colmap                 # Debian / Ubuntu
```

You can also build COLMAP from source and point `$COLMAP_BIN` or `--colmap` at the executable. The
results above were produced with COLMAP 4.2.0.dev0, CPU only. A GPU build (`--gpu`) speeds up feature
extraction and matching.

## Usage

```bash
asteroid-colmap run -w work              # download → prepare → reconstruct → catalog → plot
```

The same steps one at a time:

```bash
asteroid-colmap download    -w work      # 130 FIT + LBL (≈ 540 MB) and work/metadata.csv
asteroid-colmap prepare     -w work      # PNGs, feature masks, automatic orientation check
asteroid-colmap reconstruct -w work      # COLMAP (label-poses mode); --mode incremental
asteroid-colmap catalog     -w work      # work/catalog/*.csv, *.ply, summary.json
asteroid-colmap plot        -w work      # work/plots/*.png + pointcloud.html
asteroid-colmap info        -w work      # dataset, camera model and the latest summary
```

These options are the most useful (`asteroid-colmap <step> -h` lists the rest):

| option | step | effect |
|---|---|---|
| `--max-images N` | download | evenly sub-sample N images, for a quick test run |
| `--subdirs 2011205_RC3` | download | one sequence only |
| `--flip auto\|none\|ud\|lr\|rot180` | prepare | array orientation (auto picks `ud` for these FITS files) |
| `--mode label-poses\|incremental` | reconstruct | mapping strategy, see below |
| `--no-refine-poses` | reconstruct | keep the label poses fixed and only triangulate |
| `--matching pairs\|exhaustive\|sequential` | reconstruct | `pairs` (default) matches views less than `--max-view-angle` (40°) apart |
| `--skip-features` | reconstruct | reuse the feature database and re-run mapping only |
| `--spacing 10` | catalog | cell size (km) of the curated subset |

On an Apple-silicon laptop with CPU-only COLMAP, feature extraction takes 0.8 min, matching 5.2 min, and
mapping 35 s in label-poses mode or 4.7 min in incremental mode. A workspace takes about 1 GB: the raw
FITS are 540 MB, the COLMAP database 240 MB, and the catalog 170 MB, most of it `observations.csv`.

## Pipeline

1. **download** reads the PDS directory listings and fetches the FIT and LBL files for the chosen
   filters in parallel. It then parses every PDS3 label into `metadata.csv` (time, quaternion,
   spacecraft vector, Sun vector, sub-spacecraft point, phase). The Vesta rotation model is checked
   against the label sub-spacecraft points, and agrees to better than 0.001°.
2. **prepare** stretches the calibrated radiance to 8-bit inside the body. It writes a feature mask
   that is eroded 8 px from the limb and the terminator, so that SIFT does not fire on the silhouette.
   It also detects the array orientation by comparing the observed lit disk with the disk predicted
   from the label geometry. The stored FITS rows are bottom-up, so `ud` is selected
   ([figure 03](examples/vesta_rc3/plots/03_orientation_check.png)).
3. **reconstruct** extracts SIFT features with the intrinsics held fixed. The OPENCV model comes from
   the instrument kernel: fx = 10716.2 px, fy = 10723.1 px, k1 = 0.189. Image pairs are chosen from the
   label viewing geometry, then matched with guided matching. Mapping uses one of the two modes below.
4. **catalog** fits a robust similarity transform (Umeyama, outlier-trimmed) from the COLMAP camera
   centres to the label spacecraft positions in the body-fixed frame. It converts every point to
   km, latitude/longitude and height above the ellipsoid, then grades the points, picks a curated
   subset, and writes each observation in three pixel conventions.
5. **plot** writes 11 figures and an interactive Plotly point cloud.

### Mapping modes

`label-poses` (default) seeds a COLMAP model with the PDS label poses and alternates
`point_triangulator` (poses fixed) with `bundle_adjuster` (poses and points refined), for two rounds.
This mode has three advantages:

- The model is born in the body-fixed frame and in km.
- It needs no initial image pair. With a 5.5° field of view the views are nearly affine, so two-view
  geometry is weakly conditioned.
- It is 8× faster.

`incremental` runs COLMAP's standard `mapper` from scratch without any prior poses. The result is
tied to the body frame afterwards, by the same similarity fit used in the catalog step. This makes it
an independent check on both the label pointing and the label-poses catalog.

> **Narrow-angle pitfall.** COLMAP rejects any camera whose focal length is more than
> `Mapper.max_focal_length_ratio` (default 10) times the image size. It does this silently: the mapper
> reports "No good initial image pair found", and the triangulator returns an empty model. The Dawn FC
> ratio is 10.47, so every mapping call here raises the limit to 2 × the camera's ratio. Any
> narrow-angle camera, for example an OSIRIS-REx PolyCam, NEAR MSI or Hera AFC, hits the same wall.

## Catalog files

All files are written to `<workdir>/catalog/`. Example copies from the RC3 run are in
[examples/vesta_rc3/](examples/vesta_rc3/). The two largest files are not included there.

| file | content |
|---|---|
| `landmarks.csv` | every point with track ≥ 3: `landmark_id`, `x/y/z_km`, `lat/lon_deg`, `radius_km`, `height_km`, `track_length`, `reproj_error_px`, `max_tri_angle_deg`, `mean_range_km`, `gsd_km`, `gray`, `grade`, `outlier` |
| `landmarks_curated.csv` | the best grade A/B landmark per equal-area 10 km cell (7,509 for RC3) |
| `observations.csv` | one row per (landmark, image), described below |
| `cameras.csv` | per image: geometry from the label, registration, landmark count, reprojection RMS, and position, attitude and boresight residuals relative to the label |
| `landmarks.ply` | point cloud in km, body-fixed (for MeshLab or CloudCompare) |
| `summary.json` | counts, alignment statistics, the similarity transform, and the COLMAP version |

Each row of `observations.csv` holds:

- `u, v`: COLMAP pixel coordinates in the oriented PNG; the image corner is 0 and the first pixel centre is 0.5.
- `sample_ik, line_ik`: 0-based pixel-centre coordinates in the instrument-kernel frame.
- `fits_col, fits_row`: the same point as indices into the stored FITS array.
- `residual_u/v_px`: reprojection residuals.
- `label_pred_du/dv_px`: the offset from the position predicted by the label pose.
- `emission_deg`, `incidence_deg`.

**Frames.**

- Positions are in Vesta's IAU 2015 "Claudia double-prime" body-fixed frame:
  pole RA/Dec 309.031°/42.235°, W0 = 285.39°, planetocentric latitude, east-positive longitude in
  [0°, 360°). This is the frame the Dawn FC PDS labels use. The older `dawn_vesta_v04.tpc` "Claudia"
  frame has W0 = 75.39°, which is off by 210°.
- Landmark grades:
  - **A**: track ≥ 8, angle ≥ 15°, error ≤ 1 px.
  - **B**: track ≥ 4, angle ≥ 5°, error ≤ 2 px.
  - **C**: everything else.
- `outlier` means |height| > 60 km.

## Figures

| | |
|---|---|
| `01_montage` input frames, oriented | `07_pointcloud` 3-D landmarks coloured by height |
| `02_geometry` sub-spacecraft track, phase and range | `08_landmark_map` equirectangular map with the curated set |
| `03_orientation_check` predicted vs observed lit disk per flip | `09_quality` track length, angle and error with the grade thresholds |
| `04_features` triangulated keypoints on the best images | `10_radius_vs_latitude` height above the ellipsoid (Rheasilvia in the south) |
| `05_covisibility` landmarks shared by each image pair | `11_landmark_chips` image patches of grade A landmarks across views |
| `06_alignment` residuals between COLMAP and the label poses | `pointcloud.html` interactive Plotly point cloud with the cameras |

<p align="center">
  <img src="examples/vesta_rc3/plots/07_pointcloud.png" width="49%" alt="Point cloud">
  <img src="examples/vesta_rc3/plots/06_alignment.png" width="49%" alt="Alignment residuals">
</p>

## Validation: two independent mapping modes

Both modes start from the same feature database, so their landmarks can be matched through the
keypoints they share:

```bash
mkdir -p work_inc/colmap
cp -R work/images work/masks work/metadata.csv work/prepare.json work_inc/
sqlite3 work/colmap/database.db ".backup work_inc/colmap/database.db"
asteroid-colmap reconstruct -w work_inc --mode incremental --skip-features
asteroid-colmap catalog     -w work_inc
asteroid-colmap compare     -w work --other work_inc
```

The RC3 results are in [compare_incremental.json](examples/vesta_rc3/compare_incremental.json):

- 98,975 of about 110,000 landmarks are matched.
- The 3-D distance between matched landmarks is 117 m median and 148 m p90.
- Nearly all of that distance is a common offset of (92, −61, −43) m. After removing it, the distance
  is **15 m median and 45 m p90**.
- The height difference is 3 m median and 116 m |p90|.

Shape is therefore reproducible to a few tens of metres, a small fraction of the 490 m pixel. The
absolute position is set by how each mode is tied to the labels.

## Accuracy and caveats

- **Absolute position is about 0.1–0.2 km.** The catalog frame is tied to the PDS label poses. The
  camera centres fit those poses with an RMS of 2.3 km, and averaging that over 130 images leaves
  roughly 0.2 km of frame uncertainty. This matches the 0.12 km offset between the two modes.
- **Lateral position and pointing are degenerate.** A 5.5° camera at 5,500 km cannot tell a 2 km
  lateral shift of the spacecraft from a 0.02° (about 80″) pointing change. The 2.26 km position RMS
  and the 79″ attitude median are the same effect. Treat the refined camera poses as a consistent pair,
  not as independent estimates.
- **Coverage is limited by lighting.** Only the sunlit southern and equatorial terrain is mapped; the
  north was in polar night. Later Dawn phases (Survey, HAMO, LAMO) fill the gaps, and the archive
  layout and labels are the same.
- **Landmarks are SIFT points, not SPC maplets.** Each landmark is a 3-D point with a track; there is
  no local topography or albedo patch. `11_landmark_chips` shows what each one looks like.
- `--max-view-angle` (40°) and the 8-px mask erosion are tuned for this approach geometry. Closer
  phases need smaller angles or `--matching exhaustive`.

## Other datasets

A new phase or target needs one `Dataset` entry in
[src/asteroid_colmap/config.py](src/asteroid_colmap/config.py): the archive URL, sub-directories,
body rotation model and ellipsoid, and camera key. Ceres RC3 (`DWNCFC2_1B`) uses the same camera,
label keywords and directory layout. A different camera also needs a `FramingCamera` entry in
[camera.py](src/asteroid_colmap/camera.py).

## Why COLMAP rather than NASA GIANT

[docs/ENGINE_EVALUATION.md](docs/ENGINE_EVALUATION.md) compares the two engines. In short, COLMAP
builds landmarks from images alone. GIANT's landmark navigation (SFN), and its template mode of
constraint matching, start from a shape model or DEM that this dataset does not yet have. GIANT is
the natural *consumer* of a catalog like this one, for relative OpNav, star-based attitude and camera
calibration. It is not the tool that builds the catalog.

## Tests

```bash
pytest -q        # 37 tests, < 1 s, no network and no COLMAP needed
```

The tests cover:

- PDS3 label parsing on a real RC3 label;
- quaternion and rotation conventions;
- the Vesta rotation model, checked against the label's sub-spacecraft point to 1e-3°;
- the camera projection round trip and pixel-frame conversions;
- COLMAP TXT model I/O;
- the label-pose seed model;
- pair selection;
- landmark grading, equal-area cells and catalog comparison.

## Credits

- **Data:** Dawn FC2 calibrated images, PDS Small Bodies Node
  (<https://sbnarchive.psi.edu/pds3/dawn/fc/>). Sierks et al. (2011), *The Dawn Framing Camera*,
  Space Sci. Rev. 163, 263–327.
- **Rotation model:** Archinal et al. (2018), IAU WGCCRE 2015.
- **Ellipsoid:** Russell et al. (2012), Science 336, 684.
- **COLMAP:** Schönberger & Frahm (2016), *Structure-from-Motion Revisited*, CVPR.
