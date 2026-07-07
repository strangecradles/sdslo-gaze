import sys

sys.path.insert(0, "src")

import numpy as np

from sdslo_gaze import dynamics
from sdslo_gaze.microsaccade import (
    Microsaccade,
    detect,
    duty_cycle_report,
    main_sequence,
    noise_floor_arcmin,
    surrogate_null,
)
from sdslo_gaze.scan_timing import MeasurementState

FS = 1000.0
DUR_S = 2.0
N = int(DUR_S * FS)

# Ground truth for the four injected microsaccades: amplitude follows the expected
# main sequence in direction (dynamics.main_sequence_peak_velocity is monotonically
# increasing with amplitude -- larger events really are faster), but the *duration* used
# to build each waveform is fixed at ~15 ms (the middle of the 6-30 ms microsaccade band)
# rather than derived from the main-sequence peak-velocity curve. A minimum-jerk profile
# of the given amplitude and duration is used instead, which is self-consistent (peak
# velocity = 1.875 * amplitude / duration) and reproduces the qualitative main-sequence
# trend the detector and main_sequence() are tested against.
_AMPS_ARCMIN = (6.0, 12.0, 20.0, 30.0)
_ONSETS_S = (0.3, 0.8, 1.3, 1.7)
_DIRECTIONS_RAD = (0.4, 2.1, -1.0, 2.7)
_EVENT_DUR_S = 0.015


def _minjerk(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk 0->1 profile (smooth, zero velocity/accel at both ends)."""
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def _synthetic_trace(seed: int = 0):
    """~1 kHz, 2 s trace: OU pursuit drift + 4 injected microsaccades of known geometry.

    Returns (t, x, y, truth) where truth is a list of dicts with the injected
    amplitude, onset/offset time, and duration for each event.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / FS
    t = np.arange(N) * dt

    # Slow OU-ish pursuit/fixational drift (steady-state velocity std ~3 arcmin/s).
    tau = 0.3
    sigma_v = 3.0
    phi = np.exp(-dt / tau)
    innov = sigma_v * np.sqrt(max(1.0 - phi ** 2, 0.0))
    vx = np.zeros(N)
    vy = np.zeros(N)
    for i in range(1, N):
        vx[i] = vx[i - 1] * phi + rng.normal(0.0, innov)
        vy[i] = vy[i - 1] * phi + rng.normal(0.0, innov)
    x = np.cumsum(vx) * dt
    y = np.cumsum(vy) * dt

    # Small registration/measurement noise.
    x = x + rng.normal(0.0, 0.15, N)
    y = y + rng.normal(0.0, 0.15, N)

    truth = []
    for amp, t0, ang in zip(_AMPS_ARCMIN, _ONSETS_S, _DIRECTIONS_RAD):
        i0 = int(round(t0 * FS))
        i1 = i0 + int(round(_EVENT_DUR_S * FS))
        dxv, dyv = amp * np.cos(ang), amp * np.sin(ang)
        prof = _minjerk(np.linspace(0.0, 1.0, i1 - i0))
        x[i0:i1] += dxv * prof
        y[i0:i1] += dyv * prof
        x[i1:] += dxv
        y[i1:] += dyv
        truth.append({
            "amp": amp, "t0": float(t[i0]), "t1": float(t[i1 - 1]),
            "dur": float(t[i1 - 1] - t[i0]),
        })
    return t, x, y, truth


def _match_nearest(events, t0: float):
    return min(events, key=lambda e: abs(e.t_onset - t0))


def test_detect_finds_injected_microsaccades():
    t, x, y, truth = _synthetic_trace(seed=0)
    events = detect(t, x, y)

    assert 3 <= len(events) <= 6

    for g in truth:
        e = _match_nearest(events, g["t0"])
        assert abs(e.t_onset - g["t0"]) < 0.01
        assert abs(e.amp_arcmin - g["amp"]) / g["amp"] < 0.25
        assert 0.006 <= e.dur_s <= 0.03


def test_detect_respects_observed_mask_no_false_gap_events():
    t, x, y, truth = _synthetic_trace(seed=1)
    observed = np.ones(N, dtype=bool)
    # Carve out a "flyback" gap that straddles nothing but a quiet stretch of drift,
    # and also directly abuts one injected event's tail so detect() must not fuse
    # the gap edge into a spurious event.
    gap_lo, gap_hi = int(0.5 * FS), int(0.6 * FS)
    observed[gap_lo:gap_hi] = False

    events = detect(t, x, y, observed=observed)
    assert all(e.fully_observed for e in events)
    # No event should claim to span the unobserved gap.
    for e in events:
        assert not (e.t_onset < t[gap_lo] < e.t_offset)
        assert not (e.t_onset < t[gap_hi - 1] < e.t_offset)


def test_main_sequence_positive_slope_high_corr():
    t, x, y, truth = _synthetic_trace(seed=0)
    events = detect(t, x, y)

    # Sanity: dynamics' main-sequence peak-velocity curve is monotone increasing,
    # i.e. the amplitudes we injected are expected to be ordered the same way in
    # peak velocity too (used here only as a qualitative cross-check, not as the
    # literal per-event target -- see the module docstring above).
    predicted = dynamics.main_sequence_peak_velocity(np.array(_AMPS_ARCMIN))
    assert np.all(np.diff(predicted) > 0)

    fit = main_sequence(events)
    assert fit["n"] >= 3
    assert fit["slope"] > 0
    assert fit["r"] > 0.8


def test_surrogate_null_collapses_detection():
    t, x, y, truth = _synthetic_trace(seed=2)
    real = detect(t, x, y)
    real_ms = main_sequence(real)
    assert real_ms["r"] > 0.9

    # Group samples into pseudo-frames (200-sample acquisition blocks); each
    # injected ~15-sample event lives inside one block, so within-block shuffling
    # destroys its waveform while preserving the per-block displacement stats.
    # Permutation typically *increases* raw event count here (it fragments each
    # clean bump into several sharp reshuffled jumps), so the decisive null-control
    # signal is the main-sequence correlation collapsing, not the raw rate -- real
    # (micro)saccades have a tight amplitude/peak-velocity coupling that survives
    # only in correctly time-ordered data.
    strip = np.arange(N) // 200
    rng = np.random.default_rng(3)
    surrogate = surrogate_null(t, x, y, strip, n_shuffle=8, rng=rng)
    surrogate_ms = main_sequence(surrogate)

    assert surrogate_ms["r"] < 0.5 * real_ms["r"]


def test_noise_floor_arcmin_recovers_injected_sigma():
    rng = np.random.default_rng(4)
    n, fs = 20000, 1000.0
    t = np.arange(n) / fs
    sigma_x, sigma_y = 0.4, 0.25
    x = rng.normal(0.0, sigma_x, n)
    y = rng.normal(0.0, sigma_y, n)

    rms_x, rms_y = noise_floor_arcmin(t, x, y, hp_hz=40.0)

    assert abs(rms_x - sigma_x) / sigma_x < 0.2
    assert abs(rms_y - sigma_y) / sigma_y < 0.2


def test_duty_cycle_report_correct_fractions_and_event_tallies():
    n_active, n_reacq, n_pred, n_missing = 700, 100, 150, 50
    state = np.concatenate([
        np.full(n_active, int(MeasurementState.OBSERVED_ACTIVE), dtype=np.int8),
        np.full(n_reacq, int(MeasurementState.OBSERVED_REACQ), dtype=np.int8),
        np.full(n_pred, int(MeasurementState.PREDICTED_FLYBACK), dtype=np.int8),
        np.full(n_missing, int(MeasurementState.MISSING_FLYBACK), dtype=np.int8),
    ])
    n_total = state.shape[0]

    def _dummy(fully_observed: bool) -> Microsaccade:
        return Microsaccade(
            t_onset=0.0, t_offset=0.01, dur_s=0.01, amp_arcmin=10.0,
            peak_vel_arcmin_s=500.0, dx_arcmin=10.0, dy_arcmin=0.0,
            direction_rad=0.0, n_samples=10, fully_observed=fully_observed,
        )

    events = [_dummy(True), _dummy(True), _dummy(False)]
    report = duty_cycle_report(state, events, sample_hz=1000.0)

    assert abs(report["observed_fraction"] - (n_active + n_reacq) / n_total) < 1e-9
    assert abs(report["predicted_fraction"] - n_pred / n_total) < 1e-9
    assert abs(report["missing_fraction"] - n_missing / n_total) < 1e-9
    assert report["n_events"] == 3
    assert report["fully_observed_events"] == 2
    assert report["gap_straddling_events"] == 1
    assert report["gap_straddling_search"] == "not_implemented"
    expected_rate = 3 / ((n_active + n_reacq) / 1000.0)
    assert abs(report["rate_hz"] - expected_rate) / expected_rate < 1e-9


def test_duty_cycle_report_gap_straddling_none_when_none_searched():
    # detect() (as fed real pipeline events) always marks fully_observed=True, so
    # gap_straddling_events must read as None (not 0) rather than implying a real
    # zero-count search was performed.
    state = np.full(100, int(MeasurementState.OBSERVED_ACTIVE), dtype=np.int8)

    def _dummy() -> Microsaccade:
        return Microsaccade(
            t_onset=0.0, t_offset=0.01, dur_s=0.01, amp_arcmin=10.0,
            peak_vel_arcmin_s=500.0, dx_arcmin=10.0, dy_arcmin=0.0,
            direction_rad=0.0, n_samples=10, fully_observed=True,
        )

    events = [_dummy(), _dummy()]
    report = duty_cycle_report(state, events, sample_hz=1000.0)

    assert report["fully_observed_events"] == 2
    assert report["gap_straddling_events"] is None
    assert report["gap_straddling_search"] == "not_implemented"
