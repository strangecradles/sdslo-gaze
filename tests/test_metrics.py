import sys

sys.path.insert(0, "src")

import numpy as np

from sdslo_gaze.metrics import (
    _detrend,
    correlate_dot,
    precision_floor,
    speed_percentiles,
    split_half_precision,
)


def test_precision_floor_recovers_noise_sigma():
    rng = np.random.default_rng(0)
    n, fs = 20000, 1000.0
    t = np.arange(n) / fs
    sigma_x, sigma_y = 2.0, 1.0
    x = rng.normal(0.0, sigma_x, n)
    y = rng.normal(0.0, sigma_y, n)

    out = precision_floor(t, x, y, hp_hz=40.0)

    assert abs(out["rms_x"] - sigma_x) / sigma_x < 0.2
    assert abs(out["rms_y"] - sigma_y) / sigma_y < 0.2


def test_split_half_precision_recovers_noise_sigma():
    rng = np.random.default_rng(1)
    n, fs = 20000, 500.0
    t = np.arange(n) / fs
    sigma = 1.5
    smooth = 5.0 * np.sin(2 * np.pi * 0.02 * t)  # slow drift, well below block cutoff
    x = smooth + rng.normal(0.0, sigma, n)
    y = smooth + rng.normal(0.0, sigma, n)

    out = split_half_precision(t, x, y, block_s=1.0)

    assert abs(out["rms_x"] - sigma) / sigma < 0.25
    assert abs(out["rms_y"] - sigma) / sigma < 0.25
    assert out["n_blocks"] > 0


def test_correlate_dot_recovers_pursuit_correlation():
    rng = np.random.default_rng(2)
    n, fs = 5000, 500.0
    t = np.arange(n) / fs
    dot_x = 100.0 * np.sin(2 * np.pi * 0.2 * t)
    dot_y = 60.0 * np.cos(2 * np.pi * 0.2 * t)
    # track = affine(dot) + small noise (display calibration + registration noise)
    track_x = 0.8 * dot_x + 3.0 + rng.normal(0.0, 0.5, n)
    track_y = 1.2 * dot_y - 2.0 + rng.normal(0.0, 0.5, n)

    out = correlate_dot(t, track_x, track_y, t, dot_x, dot_y, detrend_hz=0.05, held_out=True)

    assert out["r_x"] > 0.9
    assert out["r_y"] > 0.9
    assert out["n"] > 0
    assert out["held_out"] is True
    assert out["n_train"] == n - (int(0.6 * n) - int(0.4 * n))


def test_correlate_dot_held_out_false_uses_all_samples():
    rng = np.random.default_rng(3)
    n, fs = 3000, 300.0
    t = np.arange(n) / fs
    dot_x = 50.0 * np.sin(2 * np.pi * 0.2 * t)
    dot_y = 50.0 * np.cos(2 * np.pi * 0.2 * t)
    track_x = 1.0 * dot_x + rng.normal(0.0, 0.3, n)
    track_y = 1.0 * dot_y + rng.normal(0.0, 0.3, n)

    out = correlate_dot(t, track_x, track_y, t, dot_x, dot_y, detrend_hz=0.05, held_out=False)

    assert out["r_x"] > 0.9
    assert out["r_y"] > 0.9
    assert out["n"] == n  # detrend/highpass drops nothing here; n counts finite residuals
    assert out["held_out"] is False
    assert out["n_train"] == n


def test_correlate_dot_held_out_property_rejects_overfit_score_region():
    """held_out=True must not let a genuine held-out score be inflated by an overfit fit.

    Constructs a dot recording that is corrupted to pure noise exactly in the internal
    holdout window (the middle 20%, indices [0.4n, 0.6n)) but carries the real pursuit
    signal everywhere else; the track genuinely follows the real signal (affine + small
    noise) over the FULL duration. With held_out=True the affine is fit on the clean 80%
    and scored only against the corrupted middle -- an honest near-zero r, since that
    ground truth is unpredictable noise. With held_out=False, fit and score both use all
    samples, and the score is dominated by the 80% real correlation, so r comes out much
    higher -- the in-sample number a naive (non-held-out) evaluation would report.
    """
    rng = np.random.default_rng(5)
    n, fs = 5000, 500.0
    t = np.arange(n) / fs
    idx = np.arange(n)
    lo, hi = int(0.4 * n), int(0.6 * n)  # mirrors correlate_dot's internal holdout window
    score_region = (idx >= lo) & (idx < hi)

    true_dot_x = 80.0 * np.sin(2 * np.pi * 0.2 * t)
    true_dot_y = 60.0 * np.cos(2 * np.pi * 0.2 * t)
    # Ground-truth dot recording: real signal everywhere except a corrupted (pure-noise)
    # stretch exactly in the score region.
    dot_x = np.where(score_region, rng.normal(0.0, 50.0, n), true_dot_x)
    dot_y = np.where(score_region, rng.normal(0.0, 50.0, n), true_dot_y)
    # Track genuinely follows the real signal over the FULL duration (small noise only).
    track_x = 1.0 * true_dot_x + rng.normal(0.0, 1.0, n)
    track_y = 1.0 * true_dot_y + rng.normal(0.0, 1.0, n)

    out_held = correlate_dot(t, track_x, track_y, t, dot_x, dot_y, detrend_hz=0.05, held_out=True)
    out_all = correlate_dot(t, track_x, track_y, t, dot_x, dot_y, detrend_hz=0.05, held_out=False)

    assert out_held["held_out"] is True
    assert out_all["held_out"] is False
    assert out_held["r_x"] < out_all["r_x"]
    assert out_held["r_y"] < out_all["r_y"]
    # The honest held-out score should be near zero (the score region is unpredictable noise).
    assert abs(out_held["r_x"]) < 0.3
    assert abs(out_held["r_y"]) < 0.3
    # The in-sample (non-held-out) score is inflated by the 80% real-signal majority.
    assert out_all["r_x"] > 0.6
    assert out_all["r_y"] > 0.6


def test_speed_percentiles_constant_velocity():
    n, fs = 2000, 500.0
    t = np.arange(n) / fs
    V = 30.0  # arcmin/s
    x = V * t
    y = np.zeros(n)

    out = speed_percentiles(t, x, y, pcts=(50, 99, 99.9))

    assert abs(out["p50"] - V) / V < 0.05
    assert abs(out["p99"] - V) / V < 0.05


def test_speed_percentiles_isolates_huge_jump_in_tail():
    n, fs = 2000, 500.0
    t = np.arange(n) / fs
    V = 30.0
    x = V * t
    y_clean = np.zeros(n)
    y_jump = y_clean.copy()
    y_jump[n // 2] += 5000.0  # single unphysical jump

    baseline = speed_percentiles(t, x, y_clean, pcts=(50, 99, 99.9))
    jumped = speed_percentiles(t, x, y_jump, pcts=(50, 99, 99.9))

    assert abs(jumped["p50"] - baseline["p50"]) / baseline["p50"] < 0.1
    assert jumped["p99.9"] > baseline["p99.9"] * 10


def test_detrend_removes_ramp_preserves_fast_signal():
    n, fs = 5000, 500.0
    t = np.arange(n) / fs
    ramp = 0.5 * t
    fast = 2.0 * np.sin(2 * np.pi * 5.0 * t)
    y = ramp + fast

    out = _detrend(t, y, hz=0.5)

    edge = 200
    core = slice(edge, -edge)
    assert np.corrcoef(out[core], fast[core])[0, 1] > 0.9
    assert np.std(out) < np.std(y)


def test_detrend_preserves_nan_positions():
    n, fs = 2000, 500.0
    t = np.arange(n) / fs
    y = 0.3 * t + np.sin(2 * np.pi * 5.0 * t)
    y_nan = y.copy()
    y_nan[10:20] = np.nan

    out = _detrend(t, y_nan, hz=0.5)

    assert np.all(np.isnan(out[10:20]))
    assert np.all(np.isfinite(np.delete(out, np.arange(10, 20))))
