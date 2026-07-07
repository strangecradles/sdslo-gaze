"""Tests for the sinusoidal, no-flyback slow-axis path (desinusoid + SinusoidTiming + pipeline).

The sinusoidal MEMS slow axis is the hardware target that removes the flyback gap entirely.
These tests validate the three new pieces end to end, all on synthetic data (no data tree):

  * desinusoid resampling recovers a known spatial field from its time-uniform sinusoidal
    sampling, and a backward sweep lands on the same spatial grid as a forward one;
  * SinusoidTiming reports a gap-free ~100%-duty contract with a physically-correct
    within-sweep time map;
  * pipeline.run_sinusoid recovers injected eye motion with NO predicted/missing flyback rows.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from sdslo_gaze import desinusoid as ds
from sdslo_gaze import pipeline
from sdslo_gaze.scan_timing import MeasurementState, SinusoidTiming, observed_mask


# --- desinusoid resampling ---------------------------------------------------


def test_desinusoid_recovers_uniform_field():
    """A spatially-uniform pattern, sampled sinusoidally in time, is recovered by desinusoiding."""
    H, N, M = 24, 400, 512
    u_grid = (np.arange(N) + 0.5) / N
    target = np.sin(2 * np.pi * 3 * u_grid) + 0.4 * np.sin(2 * np.pi * 7 * u_grid)
    u_in = ds.input_spatial_coord(M)
    capture = np.tile(np.interp(u_in, u_grid, target), (H, 1))  # what the scanner acquires

    for antialias in (True, False):
        rec = ds.desinusoid_sweep(capture, N, "forward", antialias=antialias)[0]
        c = slice(int(0.1 * N), int(0.9 * N))  # score the central 80% (edges are hardest)
        assert np.corrcoef(rec[c], target[c])[0, 1] > 0.99
        assert np.sqrt(np.mean((rec[c] - target[c]) ** 2)) < 0.05


def test_backward_sweep_lands_on_forward_grid():
    """A backward sweep of the same field desinusoids to the same output as the forward sweep."""
    rng = np.random.default_rng(0)
    capture = rng.standard_normal((16, 300))
    fwd = ds.desinusoid_sweep(capture, 256, "forward", antialias=True)
    bwd = ds.desinusoid_sweep(capture[:, ::-1], 256, "backward", antialias=True)
    assert np.allclose(fwd, bwd, atol=1e-9)


def test_output_time_frac_map():
    tf_f = ds.output_time_frac(200, "forward")
    tf_b = ds.output_time_frac(200, "backward")
    assert np.all(np.diff(tf_f) > 0)                    # forward acquires low-u edge first
    assert np.allclose(tf_b, 1.0 - tf_f)                # backward is the time-reverse
    assert tf_f.min() > 0.0 and tf_f.max() < 1.0


def test_desinusoid_rejects_bad_args():
    with pytest.raises(ValueError):
        ds.desinusoid_sweep(np.zeros((4, 10)), 8, "sideways")
    with pytest.raises(ValueError):
        ds.desinusoid_sweep(np.zeros((4, 10)), 8, trim_frac=0.6)
    with pytest.raises(ValueError):
        ds.desinusoid_sweep(np.zeros(10), 8)            # not 2-D


# --- SinusoidTiming ----------------------------------------------------------


def test_sinusoid_timing_is_gap_free():
    t = SinusoidTiming(f_scan_hz=400.0, cols_per_sweep=800, trim_frac=0.1)
    assert t.sweep_rate_hz == 800.0
    assert t.sweep_period_s == pytest.approx(1.0 / 800.0)
    assert t.duty_cycle == pytest.approx(0.8)           # 1 - 2*trim
    start, end = t.flyback_interval_s(3)
    assert end == start                                 # zero-length "flyback"


def test_sinusoid_strip_time_monotonic_within_forward_sweep():
    t = SinusoidTiming(f_scan_hz=400.0, cols_per_sweep=800)
    S = 8
    strips = np.arange(800 // S)
    times = np.asarray(t.strip_time_s(np.zeros_like(strips), strips, S))  # frame 0 = forward
    assert np.all(np.diff(times) > 0)                   # time rises with column on a forward sweep
    assert times[0] >= 0.0 and times[-1] < t.sweep_period_s


def test_sinusoid_backward_sweep_reverses_time():
    t = SinusoidTiming(f_scan_hz=400.0, cols_per_sweep=800)
    S = 8
    strips = np.arange(800 // S)
    t_fwd = np.asarray(t.strip_time_s(np.zeros_like(strips), strips, S))
    # frame 1 is a backward sweep; within its own window, time falls as the column index rises.
    t_bwd = np.asarray(t.strip_time_s(np.ones_like(strips), strips, S)) - t.sweep_period_s
    assert np.all(np.diff(t_bwd) < 0)
    assert np.allclose(t_bwd, t.sweep_period_s - t_fwd)


# --- end-to-end pipeline -----------------------------------------------------


def _synthetic_sinusoid_capture(seed=0):
    """Static texture translated by a known smooth path, sampled with a sinusoidal slow axis."""
    rng = np.random.default_rng(seed)
    Wfield, H, pad = 320, 200, 60
    canvas = gaussian_filter(rng.standard_normal((H + 2 * pad, Wfield + 2 * pad)), 2.0)
    n_sweeps, M = 100, 384
    tt = np.arange(n_sweeps)
    dx = 6.0 * np.sin(2 * np.pi * tt / 80)              # smooth horizontal drift (px)
    dy = 4.0 * np.cos(2 * np.pi * tt / 100)             # smooth vertical drift (px)
    u_in = ds.input_spatial_coord(M)
    col_of_u = u_in * Wfield
    sweeps = np.empty((n_sweeps, H, M))
    for i in range(n_sweeps):
        cols = col_of_u + dx[i] + pad
        rows = np.arange(H) + dy[i] + pad
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        frame = map_coordinates(canvas, [rr, cc], order=1, mode="nearest")
        if i % 2 == 1:
            frame = frame[:, ::-1]                       # backward sweep traverses in reverse
        sweeps[i] = frame
    return sweeps, Wfield, dx, dy


def test_run_sinusoid_has_no_flyback_and_recovers_motion():
    sweeps, N, dx, dy = _synthetic_sinusoid_capture()
    res = pipeline.run_sinusoid(sweeps, f_scan_hz=400.0, cols_per_sweep=N,
                                trim_frac=0.08, out_dir=None)

    # No flyback: the gap machinery collapses on a sinusoidal capture.
    assert res.timing["scan_mode"] == "sinusoid"
    assert res.role_counts["predicted_flyback"] == 0
    assert res.role_counts["missing_flyback"] == 0
    assert res.duty["predicted_fraction"] == 0.0
    assert res.duty["missing_fraction"] == 0.0
    assert res.timing["duty_cycle"] == pytest.approx(0.84)   # 1 - 2*0.08

    # Recovered per-sweep trajectory tracks the injected motion (sign-agnostic: relative track).
    gap = res._gap
    obs = observed_mask(gap.state)
    n = len(dx)
    recx = np.array([np.median(gap.x_arcmin[obs & (gap.frame == i)])
                     if np.any(obs & (gap.frame == i)) else np.nan for i in range(n)])
    recy = np.array([np.median(gap.y_arcmin[obs & (gap.frame == i)])
                     if np.any(obs & (gap.frame == i)) else np.nan for i in range(n)])
    ok = np.isfinite(recx) & np.isfinite(recy)
    assert abs(np.corrcoef(recx[ok], dx[ok])[0, 1]) > 0.9
    assert abs(np.corrcoef(recy[ok], dy[ok])[0, 1]) > 0.9


def test_run_sinusoid_all_samples_observed_state():
    sweeps, N, _, _ = _synthetic_sinusoid_capture(seed=1)
    res = pipeline.run_sinusoid(sweeps, f_scan_hz=400.0, cols_per_sweep=N, out_dir=None)
    # Every state present is either an observed measurement or an unlocked/mislock rejection;
    # none is a flyback prediction or missing gap.
    gap = res._gap
    assert not np.any(gap.state == int(MeasurementState.PREDICTED_FLYBACK))
    assert not np.any(gap.state == int(MeasurementState.MISSING_FLYBACK))
    assert np.all(np.isnan(gap.sigma_arcmin))           # no predict-only rows -> no finite sigma
