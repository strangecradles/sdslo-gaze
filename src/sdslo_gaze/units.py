"""Calibration constants and pixel <-> arcmin <-> degree conversions.

All constants are *measured from data* in the source research repo (never guessed) and are
frozen here as the source of truth. The SD-SLO sampling is anisotropic: the fast (vertical,
row) axis and the slow (horizontal, column) axis have different pixels-per-degree, so
conversions are axis-specific.

Axis convention (SD-SLO raster frame is 808 columns x 1000 rows):
  - FAST axis  = image rows    = VERTICAL gaze   (one column is one ~85 us fast sweep)
  - SLOW axis  = image columns = HORIZONTAL gaze (one frame steps across 808 columns)

So a *vertical* gaze displacement is measured in row-pixels, and a *horizontal* gaze
displacement is measured in column-pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Measured calibration (source: calib.py / cache/khz2d_pxdeg.npy in the research repo) ---

#: Slow/horizontal axis: pixels per degree along image columns (affine fit to pursuit dot).
COL_PX_PER_DEG: float = 136.12526933374963
#: Fast/vertical axis: pixels per degree along image rows (OLS fit of atlas row vs degrees).
ROW_PX_PER_DEG: float = 124.64070866730775

#: Arcmin per pixel, per axis (60 arcmin / degree).
ARCMIN_PER_COL_PX: float = 60.0 / COL_PX_PER_DEG  # ~0.44076 '/px  (horizontal gaze)
ARCMIN_PER_ROW_PX: float = 60.0 / ROW_PX_PER_DEG  # ~0.48138 '/px  (vertical gaze)

#: Perp aliasing period of the 1-D line-scan likelihood (~1 deg); informational.
ALIAS_SPACING_ROWS: float = ROW_PX_PER_DEG
#: Photoreceptor-mosaic fine spacing (rows); informational (~2.89').
MOSAIC_SPACING_ROWS: float = 6.0

# --- SD-SLO frame geometry (canonical `test1` raster) ---
SWEEPS_PER_FRAME: int = 808  #: columns per frame == number of slow-axis steps
ROWS_PER_FRAME: int = 1000   #: rows per frame == fast-axis samples per sweep
TEST1_FPS: float = 14.633001422475107  #: measured frame rate of the canonical clip
#: Encoded (uniform-clock) line rate = fps * sweeps_per_frame. NOT the active line rate;
#: see scan_timing.py for the active/flyback correction.
ENCODED_LINE_HZ: float = TEST1_FPS * SWEEPS_PER_FRAME  # ~11823.5 Hz


def col_px_to_arcmin(px: np.ndarray | float) -> np.ndarray | float:
    """Convert horizontal (slow/column-axis) pixel displacement to arcmin."""
    return np.asarray(px) * ARCMIN_PER_COL_PX


def row_px_to_arcmin(px: np.ndarray | float) -> np.ndarray | float:
    """Convert vertical (fast/row-axis) pixel displacement to arcmin."""
    return np.asarray(px) * ARCMIN_PER_ROW_PX


def arcmin_to_deg(arcmin: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(arcmin) / 60.0


def deg_to_arcmin(deg: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(deg) * 60.0


@dataclass(frozen=True)
class Units:
    """Bundle of calibration used by a pipeline run (defaults = measured `test1` values)."""

    col_px_per_deg: float = COL_PX_PER_DEG
    row_px_per_deg: float = ROW_PX_PER_DEG
    sweeps_per_frame: int = SWEEPS_PER_FRAME
    rows_per_frame: int = ROWS_PER_FRAME
    fps: float = TEST1_FPS

    @property
    def arcmin_per_col_px(self) -> float:
        return 60.0 / self.col_px_per_deg

    @property
    def arcmin_per_row_px(self) -> float:
        return 60.0 / self.row_px_per_deg

    @property
    def encoded_line_hz(self) -> float:
        return self.fps * self.sweeps_per_frame

    def x_arcmin(self, col_px: np.ndarray | float) -> np.ndarray | float:
        """Horizontal gaze (column pixels) -> arcmin."""
        return np.asarray(col_px) * self.arcmin_per_col_px

    def y_arcmin(self, row_px: np.ndarray | float) -> np.ndarray | float:
        """Vertical gaze (row pixels) -> arcmin."""
        return np.asarray(row_px) * self.arcmin_per_row_px
