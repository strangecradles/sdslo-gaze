"""microsaccade.py — Engbert-Kliegl microsaccade detection, main sequence, and duty-cycle report.

Microsaccades are 6-30 ms ballistic movements; like all saccades they follow a *main
sequence* (peak velocity rises with amplitude). Detecting the full waveform -- not just
presence -- needs >=500 Hz sampling to put several samples on the rising flank (see
``docs/flyback.md``); this package tracks at ~960-1478 Hz, in the validated band.

``detect`` never lets a scanner flyback gap masquerade as real motion: when an
``observed`` mask is supplied it only looks for events *within* a contiguous run of
observed samples, never across a boundary into predicted/missing rows. A microsaccade
that happens to fall wholly inside the flyback gap is therefore not waveform-recoverable
from this function by construction -- its amplitude is only recoverable elsewhere as the
raw reacquisition displacement (see ``docs/flyback.md`` §4.4). Every :class:`Microsaccade`
this module returns is accordingly ``fully_observed=True``; the field is carried on the
dataclass so that events assembled by other means (e.g. a reacquisition-displacement
estimator) can be reported through the same type and tallied by :func:`duty_cycle_report`.

All inputs/outputs are plain numpy arrays (arcmin, arcmin/s, seconds). The only
cross-module dependency is :mod:`sdslo_gaze.scan_timing` (the
``MeasurementState``/``observed_mask`` contract that ``duty_cycle_report``
summarizes); ``duty_cycle_report`` takes the true acquisition rate as a
required ``sample_hz`` argument rather than defaulting to any package-wide
constant, since a wrong default silently mis-scales ``rate_hz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from .scan_timing import MeasurementState, observed_mask


@dataclass(frozen=True)
class Microsaccade:
    """One detected (micro)saccade event."""

    t_onset: float             # s
    t_offset: float            # s
    dur_s: float
    amp_arcmin: float          # net 2-D displacement, pre- to post-event bracket
    peak_vel_arcmin_s: float
    dx_arcmin: float
    dy_arcmin: float
    direction_rad: float       # atan2(dy, dx)
    n_samples: int
    fully_observed: bool       # False only for events assembled from non-waveform evidence


# --- small numeric helpers --------------------------------------------------


def _contiguous_true_runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield half-open ``(start, stop)`` index ranges of contiguous ``True`` runs."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [idx.size]))
    for s0, s1 in zip(starts, stops):
        yield int(idx[s0]), int(idx[s1 - 1]) + 1


def _ek_sigma(v: np.ndarray) -> float:
    """Engbert-Kliegl robust velocity-noise sigma: sqrt(median(v^2) - median(v)^2)."""
    return float(np.sqrt(max(np.median(v ** 2) - np.median(v) ** 2, 1e-12)))


# --- 1. detection ------------------------------------------------------------


def detect(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    *,
    vfac: float = 6.0,
    min_dur_s: float = 0.006,
    smooth_ms: float = 3.0,
    observed: np.ndarray | None = None,
) -> list[Microsaccade]:
    """Engbert-Kliegl elliptic-threshold microsaccade detection.

    Velocity is estimated per axis from a Gaussian-smoothed (sigma ``smooth_ms``)
    trace via ``np.gradient`` against the real timestamps. A per-axis threshold
    ``vfac * sqrt(median(v^2) - median(v)^2)`` is estimated once, pooled over all
    contiguous observed runs; samples with ``(vx/thx)^2 + (vy/thy)^2 > 1`` are
    marked and grouped into events lasting at least ``min_dur_s``.

    If ``observed`` is given (boolean mask, same length as ``t``), detection runs
    independently within each contiguous run of ``True`` samples: a flyback gap
    (``observed=False``) can never bridge two samples into a spurious event, and
    velocity is never computed across it.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x_arcmin, dtype=np.float64)
    y = np.asarray(y_arcmin, dtype=np.float64)
    n = t.shape[0]
    obs = np.ones(n, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)

    # Pass 1: per-run smoothed velocity, pooled to set one global EK threshold.
    run_data = []
    vx_all, vy_all = [], []
    for i0, i1 in _contiguous_true_runs(obs):
        if i1 - i0 < 3:
            continue
        tt, xx, yy = t[i0:i1], x[i0:i1], y[i0:i1]
        dt_med = float(np.median(np.diff(tt))) if tt.size > 1 else 0.0
        if not np.isfinite(dt_med) or dt_med <= 0:
            continue
        sigma_samples = max((smooth_ms / 1000.0) / dt_med, 1e-6)
        xs = gaussian_filter1d(xx, sigma_samples)
        ys = gaussian_filter1d(yy, sigma_samples)
        vx = np.gradient(xs, tt)
        vy = np.gradient(ys, tt)
        run_data.append((tt, xx, yy, vx, vy))
        vx_all.append(vx)
        vy_all.append(vy)

    if not run_data:
        return []

    thx = max(vfac * _ek_sigma(np.concatenate(vx_all)), 1e-9)
    thy = max(vfac * _ek_sigma(np.concatenate(vy_all)), 1e-9)

    events: list[Microsaccade] = []
    for tt, xx, yy, vx, vy in run_data:
        supra = (vx / thx) ** 2 + (vy / thy) ** 2 > 1.0
        m = tt.shape[0]
        i = 0
        while i < m:
            if not supra[i]:
                i += 1
                continue
            j = i
            while j < m and supra[j]:
                j += 1
            dur = float(tt[j - 1] - tt[i])
            n_samp = j - i
            if dur >= min_dur_s and n_samp >= 3:
                a, b = max(0, i - 2), min(m, j + 2)
                pre = np.array([np.median(xx[a:i + 1]), np.median(yy[a:i + 1])])
                post = np.array([np.median(xx[j - 1:b]), np.median(yy[j - 1:b])])
                d = post - pre
                events.append(Microsaccade(
                    t_onset=float(tt[i]),
                    t_offset=float(tt[j - 1]),
                    dur_s=dur,
                    amp_arcmin=float(np.hypot(d[0], d[1])),
                    peak_vel_arcmin_s=float(np.max(np.hypot(vx[i:j], vy[i:j]))),
                    dx_arcmin=float(d[0]),
                    dy_arcmin=float(d[1]),
                    direction_rad=float(np.arctan2(d[1], d[0])),
                    n_samples=int(n_samp),
                    fully_observed=True,
                ))
            i = j

    events.sort(key=lambda e: e.t_onset)
    return events


# --- 2. main sequence --------------------------------------------------------


def main_sequence(events: Sequence[Microsaccade]) -> dict:
    """Fit log10(peak velocity) ~ slope * log10(amplitude) + intercept over ``events``.

    A real (micro)saccade population shows a positive, high-correlation main
    sequence; registration noise / tracker artifacts do not. Returns
    ``{"slope", "intercept", "r", "n"}`` (all NaN if fewer than 2 usable events).
    """
    usable = [e for e in events if e.amp_arcmin > 0 and e.peak_vel_arcmin_s > 0]
    n = len(usable)
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "n": n}
    amp = np.array([e.amp_arcmin for e in usable])
    pv = np.array([e.peak_vel_arcmin_s for e in usable])
    la, lp = np.log10(amp), np.log10(pv)
    slope, intercept = np.polyfit(la, lp, 1)
    r = float(np.corrcoef(la, lp)[0, 1]) if n >= 3 and np.std(la) > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r": r, "n": n}


# --- 3. surrogate (null) control ---------------------------------------------


def surrogate_null(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    strip: np.ndarray,
    *,
    n_shuffle: int = 1,
    rng: np.random.Generator | None = None,
    **detect_kw,
) -> list[Microsaccade]:
    """Detect on a strip-permuted surrogate: the physics-grounded null control.

    ``strip`` labels each sample with its acquisition group (e.g. frame id, or a
    within-frame strip/run id); within each group the sample *values* (x, y) are
    randomly permuted while ``t`` is left untouched, destroying the temporal
    waveform but preserving the per-group displacement distribution. A real
    microsaccade population collapses under this null (detection rate drops,
    main-sequence correlation degrades); registration noise looks the same on
    both the real trace and the surrogate.

    Runs ``n_shuffle`` independent permutations and concatenates all detected
    events (divide the count by ``n_shuffle`` for an average null rate).
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x_arcmin, dtype=np.float64)
    y = np.asarray(y_arcmin, dtype=np.float64)
    strip = np.asarray(strip)

    group_idx: list[np.ndarray] = []
    for g in np.unique(strip):
        idx = np.flatnonzero(strip == g)
        if idx.size > 1:
            group_idx.append(idx)

    out: list[Microsaccade] = []
    for _ in range(max(1, n_shuffle)):
        xs, ys = x.copy(), y.copy()
        for idx in group_idx:
            perm = rng.permutation(idx.size)
            xs[idx] = x[idx][perm]
            ys[idx] = y[idx][perm]
        out.extend(detect(t, xs, ys, **detect_kw))
    return out


# --- 4. noise floor -----------------------------------------------------------


def noise_floor_arcmin(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    *,
    hp_hz: float = 40.0,
) -> tuple[float, float]:
    """Truth-free precision floor: RMS per axis above ``hp_hz`` (arcmin).

    The oculomotor band (drift, pursuit, saccades) carries ~no power above
    ``hp_hz``, so a zero-phase Butterworth high-pass isolates registration
    noise. Returns ``(rms_x, rms_y)``.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x_arcmin, dtype=np.float64)
    y = np.asarray(y_arcmin, dtype=np.float64)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("cannot estimate sample rate: no positive finite dt")
    fs = 1.0 / float(np.median(dt))
    nyq = fs / 2.0
    if hp_hz >= nyq:
        raise ValueError(f"hp_hz ({hp_hz}) must be below Nyquist ({nyq})")
    sos = butter(4, hp_hz / nyq, btype="highpass", output="sos")
    xf = sosfiltfilt(sos, x)
    yf = sosfiltfilt(sos, y)
    return float(np.sqrt(np.mean(xf ** 2))), float(np.sqrt(np.mean(yf ** 2)))


# --- 5. duty-cycle report ------------------------------------------------------


def duty_cycle_report(
    state: np.ndarray,
    events: Sequence[Microsaccade],
    *,
    sample_hz: float,
) -> dict:
    """Summarize measurement-state composition and detected events.

    ``state`` is a :class:`~sdslo_gaze.scan_timing.MeasurementState` int8 array;
    observed/predicted/missing fractions are computed with
    :func:`~sdslo_gaze.scan_timing.observed_mask` over its full length.
    ``events`` need not have come from :func:`detect` on this same array --
    ``fully_observed_events``/``gap_straddling_events`` are tallied from each
    event's ``fully_observed`` flag, so events assembled from other evidence
    (e.g. a reacquisition-displacement, amplitude-only estimate for a
    gap-straddling microsaccade) are counted correctly too.

    ``sample_hz`` is REQUIRED (no default): it must be the actual acquisition
    rate of ``state`` (e.g. the strip rate), since silently defaulting to the
    package's raw encoded line rate would be off by ~8x for a strip-cadence
    ``state`` array and would misreport ``rate_hz``.

    ``rate_hz`` divides the event count by the observed sample count over
    ``sample_hz`` for an events/observed-second estimate.

    ``detect()`` (as fed pipeline events) always marks every event
    ``fully_observed=True`` -- a flyback gap can never straddle an event by
    construction (see the module docstring) -- so ``gap_straddling_events``
    is structurally always 0 in that path. To avoid a reader misreading "0"
    as "none were searched for", this returns ``gap_straddling_events=None``
    (and ``"gap_straddling_search": "not_implemented"``) whenever no event
    has ``fully_observed=False``; real straddling events (assembled from
    other evidence, with ``fully_observed=False``) are still counted when
    present.
    """
    s = np.asarray(state, dtype=np.int8)
    n = s.shape[0]
    obs = observed_mask(s)
    n_obs = int(np.sum(obs))
    n_pred = int(np.sum(s == int(MeasurementState.PREDICTED_FLYBACK)))
    n_missing = int(np.sum(s == int(MeasurementState.MISSING_FLYBACK)))

    events = list(events)
    n_events = len(events)
    fully = sum(1 for e in events if e.fully_observed)
    straddling = n_events - fully

    observed_duration_s = n_obs / sample_hz if sample_hz > 0 else float("nan")
    rate_hz = n_events / observed_duration_s if observed_duration_s > 0 else float("nan")

    return {
        "observed_fraction": n_obs / n if n else float("nan"),
        "predicted_fraction": n_pred / n if n else float("nan"),
        "missing_fraction": n_missing / n if n else float("nan"),
        "n_events": n_events,
        "rate_hz": rate_hz,
        "fully_observed_events": fully,
        "gap_straddling_events": straddling if straddling > 0 else None,
        "gap_straddling_search": "not_implemented",
    }
