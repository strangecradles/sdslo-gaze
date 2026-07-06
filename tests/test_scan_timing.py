"""Tests for the scanner-timing / measurement-state contract (the package centerpiece)."""

import numpy as np
import pytest

from sdslo_gaze.scan_timing import (
    MeasurementState,
    ScanTiming,
    TIMING_MODELS,
    observed_mask,
)
from sdslo_gaze.units import SWEEPS_PER_FRAME, TEST1_FPS


def test_timing_models_satisfy_frame_period_invariant():
    for name in TIMING_MODELS:
        st = ScanTiming.from_model(name)
        assert abs(st.active_ms + st.flyback_ms - 1000.0 / st.fps) < 1e-3


def test_flyback10_and_active40_numbers():
    f10 = ScanTiming.from_model("flyback10ms")
    assert f10.flyback_ms == pytest.approx(10.0, abs=1e-6)
    assert f10.active_ms == pytest.approx(58.339, abs=1e-2)
    assert f10.active_line_hz == pytest.approx(13850.0, rel=1e-3)
    assert f10.duty_cycle == pytest.approx(0.854, abs=1e-2)

    a40 = ScanTiming.from_model("active40ms")
    assert a40.active_ms == pytest.approx(40.0, abs=1e-6)
    assert a40.active_line_hz == pytest.approx(20200.0, rel=1e-3)


def test_uniform_model_is_the_encoded_clock_baseline():
    u = ScanTiming.from_model("uniform")
    assert u.flyback_ms == pytest.approx(0.0)
    assert u.active_line_hz == pytest.approx(TEST1_FPS * SWEEPS_PER_FRAME, rel=1e-9)


def test_column_times_active_then_flyback_gap():
    st = ScanTiming.from_model("flyback10ms")
    # columns within a frame are strictly increasing and fit inside the active window
    t_first = st.column_time_s(0, 0)
    t_last = st.column_time_s(0, SWEEPS_PER_FRAME - 1)
    assert t_first < t_last
    assert (t_last - 0.0) * 1000.0 <= st.active_ms + 1e-6
    # a 10 ms flyback gap separates the last active column from the next frame
    gap0, gap1 = st.flyback_interval_s(0)
    assert (gap1 - gap0) * 1000.0 == pytest.approx(10.0, abs=1e-6)
    assert t_last < gap0 <= st.column_time_s(1, 0)


def test_strip_time_matches_column_midpoint():
    st = ScanTiming.from_model("flyback10ms")
    S = 8
    # strip s spans columns [s*S, s*S+S); its center column is s*S + S/2
    for s in (0, 5, 50):
        expected = st.column_time_s(0, s * S + S / 2 - 0.5)  # column_time adds +0.5
        got = st.strip_time_s(0, s, S)
        assert got == pytest.approx(expected, abs=1e-9)


def test_from_hardware_requires_exactly_one_arg():
    with pytest.raises(ValueError):
        ScanTiming.from_hardware()
    with pytest.raises(ValueError):
        ScanTiming.from_hardware(active_ms=40.0, flyback_ms=10.0)
    st = ScanTiming.from_hardware(active_ms=45.0)
    assert st.flyback_ms == pytest.approx(1000.0 / TEST1_FPS - 45.0, abs=1e-6)


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        ScanTiming(fps=TEST1_FPS, active_ms=10.0, flyback_ms=10.0)  # doesn't sum to period


def test_observed_mask_only_true_for_observed_states():
    states = np.array(
        [
            MeasurementState.OBSERVED_ACTIVE,
            MeasurementState.OBSERVED_REACQ,
            MeasurementState.PREDICTED_FLYBACK,
            MeasurementState.MISSING_FLYBACK,
            MeasurementState.REJECTED_MISLOCK,
            MeasurementState.UNLOCKED,
        ],
        dtype=np.int8,
    )
    assert observed_mask(states).tolist() == [True, True, False, False, False, False]


def test_unlocked_is_distinct_from_mislock():
    assert MeasurementState.UNLOCKED != MeasurementState.REJECTED_MISLOCK
    assert not MeasurementState.UNLOCKED.is_observed
    assert not MeasurementState.REJECTED_MISLOCK.is_observed
