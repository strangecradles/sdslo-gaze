"""Tests for the predict-only motion model and the saccadic main sequence."""

import numpy as np
import pytest

from sdslo_gaze import dynamics
from sdslo_gaze.dynamics import (
    MotionState,
    estimate_velocity,
    main_sequence_peak_velocity,
    predict_gaussian,
)


def test_main_sequence_monotonic_and_saturating():
    amps = np.array([0.5, 2.0, 6.5, 20.0, 60.0])
    v = main_sequence_peak_velocity(amps)
    assert np.all(np.diff(v) > 0)                      # strictly increasing
    assert np.all(v < dynamics.VMAX_ARCMIN_S)          # bounded by VMAX
    # at the knee amplitude A0 the velocity is VMAX*(1-1/e)
    assert main_sequence_peak_velocity(dynamics.A0_ARCMIN) == pytest.approx(
        dynamics.VMAX_ARCMIN_S * (1 - np.exp(-1)), rel=1e-9
    )


def test_predict_only_moves_and_grows_uncertainty():
    ms = MotionState.from_observations(pos=[0.0, 0.0], vel=[600.0, 0.0], pos_var=1.0)
    p = predict_gaussian(ms, dt=0.010)  # a 10 ms flyback
    assert p.pos[0] > 0.0                     # moved in the velocity direction
    assert p.pos[1] == pytest.approx(0.0)     # no y velocity -> no y motion
    assert np.all(p.pos_var > ms.pos_var)     # uncertainty grew (no measurement)


def test_predict_only_velocity_decays_toward_zero():
    ms = MotionState.from_observations(pos=[0, 0], vel=[100.0, -50.0], pos_var=0.5)
    p = predict_gaussian(ms, dt=dynamics.TAU_PURSUIT_S)  # one time constant
    assert np.all(np.abs(p.vel) < np.abs(ms.vel))        # OU mean reversion


def test_predict_zero_dt_is_identity_position():
    ms = MotionState.from_observations(pos=[3.0, -1.0], vel=[10.0, 10.0], pos_var=2.0)
    p = predict_gaussian(ms, dt=0.0)
    assert np.allclose(p.pos, ms.pos)
    assert np.allclose(p.pos_var, ms.pos_var)


def test_predict_negative_dt_raises():
    ms = MotionState.from_observations(pos=[0, 0], vel=[0, 0])
    with pytest.raises(ValueError):
        predict_gaussian(ms, dt=-0.001)


def test_estimate_velocity_recovers_constant_slope():
    t = np.linspace(0, 0.01, 8)
    v_true = np.array([500.0, -200.0])
    x = v_true[0] * t
    y = v_true[1] * t
    v = estimate_velocity(t, x, y, k=8)
    assert v == pytest.approx(v_true, rel=1e-6)


def test_estimate_velocity_handles_insufficient_data():
    assert np.allclose(estimate_velocity(np.array([0.0]), np.array([0.0]), np.array([0.0])), 0.0)
