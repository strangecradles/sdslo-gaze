"""Tests for calibration constants and px<->arcmin conversions."""

import numpy as np
import pytest

from sdslo_gaze import units


def test_anisotropic_arcmin_per_px():
    # measured calibration; horizontal (col) and vertical (row) axes differ
    assert units.ARCMIN_PER_COL_PX == pytest.approx(60.0 / 136.12526933374963, rel=1e-9)
    assert units.ARCMIN_PER_ROW_PX == pytest.approx(60.0 / 124.64070866730775, rel=1e-9)
    assert units.ARCMIN_PER_COL_PX != units.ARCMIN_PER_ROW_PX


def test_encoded_line_rate():
    assert units.ENCODED_LINE_HZ == pytest.approx(11823.465, abs=0.1)


def test_conversions_roundtrip_and_axes():
    px = np.array([0.0, 1.0, -2.5])
    assert np.allclose(units.col_px_to_arcmin(px), px * units.ARCMIN_PER_COL_PX)
    assert np.allclose(units.row_px_to_arcmin(px), px * units.ARCMIN_PER_ROW_PX)
    assert units.deg_to_arcmin(1.0) == pytest.approx(60.0)
    assert units.arcmin_to_deg(60.0) == pytest.approx(1.0)


def test_units_dataclass_matches_module_constants():
    u = units.Units()
    assert u.arcmin_per_col_px == pytest.approx(units.ARCMIN_PER_COL_PX, rel=1e-9)
    assert u.arcmin_per_row_px == pytest.approx(units.ARCMIN_PER_ROW_PX, rel=1e-9)
    assert u.encoded_line_hz == pytest.approx(units.ENCODED_LINE_HZ, rel=1e-9)
    # x uses column calibration (horizontal), y uses row calibration (vertical)
    assert u.x_arcmin(10.0) == pytest.approx(10.0 * units.ARCMIN_PER_COL_PX)
    assert u.y_arcmin(10.0) == pytest.approx(10.0 * units.ARCMIN_PER_ROW_PX)
