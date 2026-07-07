import sys

sys.path.insert(0, "src")

import numpy as np

from sdslo_gaze.gap_model import GapTrack, apply_scan_timing, sensitivity
from sdslo_gaze.scan_timing import MeasurementState, ScanTiming
from sdslo_gaze.units import TEST1_FPS

S = 8
W = 808
FPS = TEST1_FPS
N_FRAMES = 3
N_STRIPS = W // S  # 101, exact: 101 * 8 == 808


def _synthetic_track(timing: ScanTiming, *, q_value: float = 0.8):
    """A smooth, fully-in-FOV, fully-locked 3-frame strip track."""
    frames = np.repeat(np.arange(N_FRAMES, dtype=np.int64), N_STRIPS)
    strips = np.tile(np.arange(N_STRIPS, dtype=np.int64), N_FRAMES)

    true_t = np.asarray(timing.strip_time_s(frames, strips, S), dtype=np.float64)
    # Slow linear pursuit drift + a small, low-frequency sinusoid: smooth motion
    # whose implied speed is far below any physically-implausible threshold.
    x_arcmin = 5.0 * true_t + 2.0 * np.sin(2 * np.pi * 1.0 * true_t)
    y_arcmin = 3.0 * np.sin(2 * np.pi * 0.7 * true_t)

    q = np.full(frames.shape, q_value, dtype=np.float64)
    infov = np.ones(frames.shape, dtype=bool)
    # Naive encoded time; apply_scan_timing must not rely on its values.
    t_naive = frames.astype(np.float64) / FPS + strips.astype(np.float64) / (N_STRIPS * FPS)
    return {
        "t": t_naive,
        "x_arcmin": x_arcmin,
        "y_arcmin": y_arcmin,
        "q": q,
        "frame": frames,
        "strip": strips,
        "infov": infov,
    }


def _run(track: dict, timing: ScanTiming, **kwargs) -> GapTrack:
    return apply_scan_timing(
        track["t"],
        track["x_arcmin"],
        track["y_arcmin"],
        track["q"],
        track["frame"],
        track["strip"],
        S,
        timing,
        infov=track["infov"],
        **kwargs,
    )


def test_apply_scan_timing_basic_retiming_and_labeling():
    timing = ScanTiming.from_model("flyback10ms", fps=FPS, sweeps_per_frame=W)
    track = _synthetic_track(timing)
    n_input = track["frame"].shape[0]

    gt = _run(track, timing)
    assert isinstance(gt, GapTrack)

    # Every input strip stays observed: smooth motion, full quality, full FOV.
    assert int(np.sum(gt.observed())) == n_input
    # Predicted/missing flyback rows were appended -> track grew.
    assert gt.t.shape[0] > n_input

    obs_mask = gt.observed()
    for name in ("PREDICTED_FLYBACK", "MISSING_FLYBACK", "REJECTED_MISLOCK"):
        member = MeasurementState[name]
        assert not np.any(obs_mask & (gt.state == int(member)))

    obs_t = gt.t[obs_mask]
    assert np.all(np.diff(obs_t) > 0)

    for f in range(N_FRAMES):
        frame_mask = obs_mask & (gt.frame == f)
        local_t = gt.t[frame_mask] - f * timing.frame_period_s
        assert np.all(local_t <= timing.active_s + 1e-9)

    # A ~10 ms flyback interval should show up as a gap in the observed-only series.
    gaps_ms = np.diff(obs_t) * 1000.0
    assert np.any((gaps_ms > 8.0) & (gaps_ms < 15.0))


def test_unphysical_jump_is_rejected_and_excluded():
    timing = ScanTiming.from_model("flyback10ms", fps=FPS, sweeps_per_frame=W)
    track = _synthetic_track(timing)
    jump_idx = 50  # well inside frame 0, away from any frame boundary
    track["x_arcmin"][jump_idx] += 5000.0
    track["q"][jump_idx] = 0.5  # above ncc_floor: reaches the motion gate, not the lock gate

    gt = _run(track, timing)

    match = (gt.frame == track["frame"][jump_idx]) & (gt.strip == track["strip"][jump_idx])
    assert int(np.sum(match)) == 1
    assert gt.state[match][0] == int(MeasurementState.REJECTED_MISLOCK)
    assert not np.any(gt.observed() & match)


def test_bridgeable_gap_yields_finite_predicted_flyback():
    timing = ScanTiming.from_model("flyback10ms", fps=FPS, sweeps_per_frame=W)
    track = _synthetic_track(timing)

    gt = _run(track, timing, max_bridge_ms=35.0)

    pred_mask = gt.state == int(MeasurementState.PREDICTED_FLYBACK)
    assert np.any(pred_mask)
    assert np.all(np.isfinite(gt.x_arcmin[pred_mask]))
    assert np.all(np.isfinite(gt.y_arcmin[pred_mask]))
    assert not np.any(pred_mask & gt.observed())


def test_long_gap_exceeding_max_bridge_yields_missing_flyback():
    timing = ScanTiming.from_hardware(flyback_ms=40.0, fps=FPS, sweeps_per_frame=W)
    track = _synthetic_track(timing)

    gt = _run(track, timing, max_bridge_ms=35.0)

    missing_mask = gt.state == int(MeasurementState.MISSING_FLYBACK)
    assert np.any(missing_mask)
    assert np.all(np.isnan(gt.x_arcmin[missing_mask]))
    assert np.all(np.isnan(gt.y_arcmin[missing_mask]))
    assert not np.any(gt.state == int(MeasurementState.PREDICTED_FLYBACK))


def test_role_counts_and_duty_fractions_sum_correctly():
    timing = ScanTiming.from_model("flyback10ms", fps=FPS, sweeps_per_frame=W)
    track = _synthetic_track(timing)
    gt = _run(track, timing)

    counts = gt.role_counts()
    assert sum(counts.values()) == gt.t.shape[0]
    assert set(counts) == {m.label for m in MeasurementState}

    fractions = gt.duty_fractions()
    assert set(fractions) == set(counts)
    assert abs(sum(fractions.values()) - 1.0) < 1e-9


def test_sensitivity_reports_duty_fractions_per_timing_model():
    timing_ref = ScanTiming.from_model("flyback10ms", fps=FPS, sweeps_per_frame=W)
    track = _synthetic_track(timing_ref)

    out = sensitivity(
        track["t"],
        track["x_arcmin"],
        track["y_arcmin"],
        track["q"],
        track["frame"],
        track["strip"],
        S,
        fps=FPS,
        sweeps_per_frame=W,
        infov=track["infov"],
    )

    assert set(out) == {"active40ms", "flyback10ms"}
    for fractions in out.values():
        assert abs(sum(fractions.values()) - 1.0) < 1e-9
        assert fractions["observed_active"] + fractions["observed_reacq"] > 0.0
