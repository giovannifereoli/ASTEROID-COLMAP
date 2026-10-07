# Engine evaluation: COLMAP vs NASA GIANT

**Question.** Which engine should build a landmark catalog of Vesta from the 130 Dawn FC2 RC3/RC3B
approach images: [COLMAP](https://github.com/colmap/colmap) or
[NASA GIANT](https://github.com/nasa/giant)?

**Verdict: COLMAP builds the catalog, and GIANT is what uses it.** GIANT's landmark machinery
assumes the landmarks or the shape model already exist; this dataset has neither. COLMAP builds the
landmarks from the images alone. The two tools are complementary, and the recommended workflow at
the end uses both.

## What each tool is

**COLMAP** is a general structure-from-motion and multi-view-stereo system. It is written in C++,
driven here through its command-line interface, and BSD-licensed. It matches SIFT features across
images, links them into multi-view tracks, and estimates the 3-D points and camera poses together
with bundle adjustment. This evaluation used version 4.2.0.dev0 (CPU build).

**GIANT** (Goddard Image Analysis and Navigation Tool) is NASA Goddard's optical-navigation library,
developed for OSIRIS-REx. It is written in Python with Cython and is Apache-2.0 licensed. Version
2.0.0 was released on 3 September 2025. Its main parts are:

- `stellar_opnav`: attitude from star fields, using the Gaia catalog.
- `calibration`: geometric camera calibration and alignment.
- `relative_opnav`: target-relative navigation, with estimators for cross-correlation, limb and
  ellipse matching, moments, unresolved targets, **SFN (surface feature navigation)** and, new in
  2.0, **constraint matching**.
- `ray_tracer`: KD-tree shapes and illumination.
- Coverage, photometry, and stereophotoclinometry (SPC) interfaces: `Maplet` and
  `spc_to_feature_catalog`.

The repository even includes a Dawn example, `examples/dawn_giant`. It uses Vesta approach OpNav
images together with SPICE kernels.

## Comparison

| criterion | COLMAP | GIANT |
|---|---|---|
| Builds landmarks from images alone | **Yes.** Multi-view tracks and bundle adjustment; RC3 gives 94,576 landmarks (7,487 curated) | **No.** See the note on GIANT's landmark modes below this table |
| Prior information needed | Intrinsics only. The label poses serve as an optional seed and as the datum for the frame | A shape model or DEM for SFN, templates, limbs and ray tracing; SPICE for geometry |
| Narrow-angle camera (5.5° FOV, f/w = 10.47) | Works after raising `Mapper.max_focal_length_ratio`. Otherwise COLMAP silently triangulates nothing | Native. Built for spacecraft cameras, with OpenCV, Owen and Brown models |
| Output | Sparse model (cameras, images, points3D); this package turns it into a body-fixed catalog | Navigation measurements and residuals, star-based attitude, calibrated camera models |
| Runtime on RC3 (CPU) | 0.8 min features, 3.1 min matching, 30 s mapping (label poses) or 3.8 min (incremental) | Not measured: the import fails, see the install log |
| Install on macOS arm64 with Python 3.14 | Prebuilt binary (`brew install colmap`, conda-forge, apt) | `pip install` from git failed in three successive ways (see the install log) |
| Used in flight | Common in planetary science research; not a flight OpNav tool | OSIRIS-REx OpNav heritage |

GIANT has two ways to work with surface landmarks, and neither builds a catalog from scratch:

- **SFN** needs an existing feature catalog of SPC maplets or DEM tiles. The repository's
  `dem_to_landmarks.py` example starts from a DEM.
- **Constraint matching** has two variants.
  - *Image to image*: in GIANT's own words, the observations "are not tied to known points on the
    surface", and they constrain only the relative motion of the camera, up to scale.
  - *Image to template*: the template is rendered from a shape model, and each observation "will be
    slightly different", so no persistent multi-view tracks form.

### GIANT install log (v2.0.0f, isolated venv, macOS 26 arm64, Python 3.14)

1. **Build failure: `omp.h` not found.** Apple clang ships without OpenMP. Building against
   Homebrew's `libomp` (setting `CFLAGS`, `CPPFLAGS` and `LDFLAGS`) then compiles the Cython
   extensions successfully.
2. **Import failure: `No module named 'cv2'`.** OpenCV is listed in `environment.yml`, but it is not
   a pip dependency. Installing `opencv-python-headless` fixes this step.
3. **Import failure: `RoMa is not installed`.** `giant.image_processing.feature_matchers` imports
   its RoMa matcher unconditionally. RoMa is a PyTorch dense matcher that must be installed from its
   GitHub repository, so `giant.image_processing` and `giant.relative_opnav`, which includes SFN and
   constraint matching, cannot be imported without PyTorch and RoMa.

The supported route is probably the conda `environment.yml`, plus RoMa on top. That route can work,
but the dependency stack is much heavier than a single COLMAP binary, and it does not remove the
shape-model requirement.

## What was done

- **COLMAP:** the full pipeline in this package. It was run in two independent mapping modes, which
  agree to 16 m median and 46 m p90 after removing a common 0.10 km offset; see the README.
  - Pitfall found: COLMAP's default focal-length-ratio limit of 10 is below the FC's 10.47. Above that
    limit COLMAP discards the camera as bogus without raising an error.
  - Pitfall found: with DSP-SIFT, COLMAP's covariant extractor ignores
    `SiftExtraction.max_num_orientations`. 18 % of the RC3 keypoints were second-orientation copies
    at the same pixel, and they turned single surface features into duplicate landmarks. The
    package removes the copies from the database after extraction.
- **GIANT:** a review of the repository, including the changelog, module tree, the
  constraint-matching and SFN sources, the SPC utilities and the Dawn example. Three install attempts
  followed, as logged above.

## When GIANT is the better choice

- **Navigation against an existing map.** Vesta already has Dawn-era SPC shape models and DTMs.
  Working against such a map is exactly what SFN and template constraint matching do.
- **Attitude independent of the spacecraft's own estimate.** GIANT's stellar OpNav measures it from
  star fields. This package can only refine the label attitude through bundle adjustment.
- **Camera calibration** from star fields, and **center-finding or limb-based OpNav** at long range,
  where the body covers few pixels.
- **Measurements for orbit determination,** with flight-proven measurement models.

## Recommended workflow

1. **This package (COLMAP)** produces sparse landmarks with tracks, refined camera poses, and residuals
   relative to the labels.
2. **Densify.** COLMAP's `patch_match_stereo` needs CUDA. Alternatively, run SPC around the curated
   landmarks to produce local DEMs, or maplets.
3. **GIANT** turns those into a feature catalog (`Maplet`, `spc_to_feature_catalog`) and runs SFN or
   template constraint matching on later phases (Survey, HAMO). Its stellar OpNav and calibration
   modules can also check the attitude and the camera model independently.

A direct exporter from `landmarks_curated.csv` to a GIANT feature catalog would bridge steps 1 and 3.
It is not implemented yet.
