"""Landmark catalogs of small bodies from Dawn Framing Camera images with COLMAP.

Pipeline stages (each is also a CLI sub-command of ``asteroid-colmap``):

1. :mod:`~asteroid_colmap.download`   - fetch calibrated FITS + PDS3 labels from the PDS SBN archive
2. :mod:`~asteroid_colmap.preprocess` - FITS -> 8-bit PNG, body masks, image-orientation check
3. :mod:`~asteroid_colmap.reconstruct` - COLMAP feature extraction, matching and incremental mapping
4. :mod:`~asteroid_colmap.georef` / :mod:`~asteroid_colmap.catalog` - tie the SfM model to the
   body-fixed frame and write the landmark catalog
5. :mod:`~asteroid_colmap.plots`      - figures
6. :mod:`~asteroid_colmap.ncc`        - find the catalog landmarks in new images by normalised
   cross-correlation and correct their poses (``templates`` and ``match``)
7. :mod:`~asteroid_colmap.pose`       - relative position and attitude of the camera from the
   matches, and a smoothed arc per sequence (``pose``)
"""

__version__ = "0.1.0"
