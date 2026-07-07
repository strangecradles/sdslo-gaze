"""metrics.py — truth-optional precision, correlation, and speed measures.

These functions score a 2-D gaze trace (plain numpy arrays, arcmin units) without
requiring any ground truth:

- ``precision_floor``: registration-noise RMS above the oculomotor band (>~40 Hz).
- ``split_half_precision``: repeatability-style noise RMS from interleaved samples
  after removing slow (smooth) signal content.
- ``correlate_dot``: correlation vs a pursuit stimulus, with an honest held-out
  affine-calibration split so the reported r is not an in-sample fit artifact.
- ``speed_percentiles``: instantaneous-speed distribution, used to surface
  unphysical jumps caused by naive/uniform timing assumptions.

All inputs/outputs are plain numpy arrays; no dependency on the rest of the
package beyond ``numpy``/``scipy``. Time arrays are assumed near-uniformly
sampled (the sampling rate is estimated from the median time step).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


# --- small numeric helpers -------------------------------------------------


def _sample_rate(t: np.ndarray) -> float:
    """Estimate the sampling rate (Hz) from the median positive time step."""
    t = np.asarray(t, dtype=np.float64)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("cannot estimate sample rate: no positive finite dt")
    return float(1.0 / np.median(dt))


def _fill_nan(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in ``y`` against ``t`` (constant extrapolation)."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if m.all() or not m.any():
        return y.copy()
    return np.interp(t, t[m], y[m])


def _butter_sos(fs: float, hz: float, btype: str, order: int = 2):
    nyq = fs / 2.0
    wn = min(max(hz / nyq, 1e-6), 0.999)
    return butter(order, wn, btype=btype, output="sos")


def _filtered(t: np.ndarray, y: np.ndarray, hz: float, btype: str, order: int = 2) -> np.ndarray:
    """Apply a zero-phase Butterworth filter, robust to NaNs.

    NaNs are filled (via linear interpolation) only to make the filter
    well-defined; the returned array preserves NaN at the original NaN
    positions.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if int(m.sum()) < 8:
        return np.full_like(y, np.nan)
    fs = _sample_rate(t)
    filled = _fill_nan(t, y)
    sos = _butter_sos(fs, hz, btype, order=order)
    out = sosfiltfilt(sos, filled)
    return np.where(m, out, np.nan)


def _contiguous_runs_by_dt(t: np.ndarray, max_gap: float) -> list[tuple[int, int]]:
    """Split ``t`` into half-open ``(start, stop)`` index ranges of contiguous "small-step" runs.

    A run boundary is placed wherever the step to the next sample exceeds
    ``max_gap`` seconds (e.g. a frame-boundary flyback step compressed into a
    single sample-step by removed rows). Each returned range spans at least
    one sample; consecutive samples within a range have ``dt <= max_gap``.
    """
    t = np.asarray(t, dtype=np.float64)
    n = t.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]
    dt = np.diff(t)
    breaks = np.flatnonzero(~(np.isfinite(dt) & (dt <= max_gap)))
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [n]))
    return [(int(s0), int(s1)) for s0, s1 in zip(starts, stops)]


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite ** 2)))


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    aa = a[m] - a[m].mean()
    bb = b[m] - b[m].mean()
    denom = float(np.sqrt(np.sum(aa ** 2) * np.sum(bb ** 2)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def _affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares fit y ~= a*x + b; returns (a, b), (nan, nan) if ill-posed."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 4 or float(np.std(x[m])) <= 0:
        return float("nan"), float("nan")
    a = np.column_stack([x[m], np.ones(int(m.sum()))])
    coef, *_ = np.linalg.lstsq(a, y[m], rcond=None)
    return float(coef[0]), float(coef[1])


def _apply_affine(x: np.ndarray, coef: tuple[float, float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not (np.isfinite(coef[0]) and np.isfinite(coef[1])):
        return np.full_like(x, np.nan)
    return coef[0] * x + coef[1]


def _detrend(t: np.ndarray, y: np.ndarray, hz: float) -> np.ndarray:
    """Subtract a low-pass (below ``hz``) trend from ``y``.

    Robust to NaNs: the trend is estimated from the finite samples only (via
    interpolation to fill gaps before filtering), but the returned residual
    keeps NaN at every position where ``y`` was NaN.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if int(m.sum()) < 8:
        return y - float(np.nanmean(y)) if m.any() else y.copy()
    trend = _filtered(t, y, hz, "lowpass")
    return y - trend


# --- public API -------------------------------------------------------------


def precision_floor(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    *,
    hp_hz: float = 40.0,
) -> dict:
    """Truth-free registration-noise floor.

    High-passes each axis above ``hp_hz`` (the eye contributes ~no power up
    there), and reports the residual RMS per axis in arcmin. This is the
    ~microsaccade-detection floor.

    The observed samples typically have flyback rows removed, so a raw
    frame-boundary time step is a large gap compressed into a single
    sample-step; high-passing the whole array as if uniformly sampled turns
    each such boundary into a transient that inflates the noise floor. To
    avoid this, the trace is split into contiguous runs whose consecutive
    ``dt`` is "small" (<= 3x the overall median ``dt``) via
    :func:`_contiguous_runs_by_dt`; each run is high-passed independently
    (runs too short for the filter are skipped) and the residuals are pooled
    (concatenated) before computing the per-axis RMS.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x_arcmin, dtype=np.float64)
    y = np.asarray(y_arcmin, dtype=np.float64)

    dt_all = np.diff(t)
    dt_all = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    if dt_all.size == 0:
        return {"rms_x": float("nan"), "rms_y": float("nan")}
    max_gap = 3.0 * float(np.median(dt_all))

    resid_x: list[np.ndarray] = []
    resid_y: list[np.ndarray] = []
    for i0, i1 in _contiguous_runs_by_dt(t, max_gap):
        tt, xx, yy = t[i0:i1], x[i0:i1], y[i0:i1]
        if tt.size < 2:
            continue
        try:
            hx = _filtered(tt, xx, hp_hz, "highpass")
            hy = _filtered(tt, yy, hp_hz, "highpass")
        except ValueError:
            continue  # run too short for the filter's padlen; skip it
        resid_x.append(hx)
        resid_y.append(hy)

    if not resid_x:
        return {"rms_x": float("nan"), "rms_y": float("nan")}
    return {
        "rms_x": _rms(np.concatenate(resid_x)),
        "rms_y": _rms(np.concatenate(resid_y)),
    }


def split_half_precision(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    *,
    block_s: float = 1.0,
) -> dict:
    """Repeatability-style precision floor from interleaved samples.

    Removes signal content that varies slower than ``block_s`` (a low-pass
    trend below ``1 / (2 * block_s)`` Hz), then splits the residual into two
    interleaved (even/odd sample) series. Since the interleaved series share
    the same underlying (now-removed) smooth signal but independent noise,
    the RMS of their difference divided by sqrt(2) estimates the per-sample
    noise sigma.
    """
    t = np.asarray(t, dtype=np.float64)
    cutoff_hz = 1.0 / (2.0 * block_s) if block_s > 0 else float("inf")
    hx = _filtered(t, x_arcmin, cutoff_hz, "highpass")
    hy = _filtered(t, y_arcmin, cutoff_hz, "highpass")

    def _interleaved_rms(h: np.ndarray) -> float:
        finite = h[np.isfinite(h)]
        if finite.size < 4:
            return float("nan")
        a, b = finite[0::2], finite[1::2]
        n = min(a.size, b.size)
        if n == 0:
            return float("nan")
        diff = a[:n] - b[:n]
        return float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(2.0))

    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    n_blocks = max(1, int(round(duration / block_s))) if block_s > 0 else 1
    return {
        "rms_x": _interleaved_rms(hx),
        "rms_y": _interleaved_rms(hy),
        "n_blocks": n_blocks,
    }


def correlate_dot(
    t_track: np.ndarray,
    x_track: np.ndarray,
    y_track: np.ndarray,
    t_dot: np.ndarray,
    x_dot: np.ndarray,
    y_dot: np.ndarray,
    *,
    detrend_hz: float = 0.05,
    held_out: bool = True,
) -> dict:
    """Correlate a tracked trace against the pursuit dot, honestly.

    The dot is resampled onto the track's time base, both series are
    detrended below ``detrend_hz`` (removing drift while keeping the pursuit
    band and faster motion), and a per-axis affine (display calibration) is
    fit on a TRAIN split. Pearson r and RMS (arcmin) are reported on a
    held-out middle block (``held_out=True``) or on all samples otherwise.

    The returned dict always carries ``"held_out"`` (True only when a real
    disjoint train/score split was actually used -- i.e. ``held_out=True``
    AND ``n_total >= 40``; otherwise the score is in-sample) and
    ``"n_train"`` (the size of the train split), so a caller can't mistake an
    in-sample fallback score for a genuine held-out one.
    """
    t_track = np.asarray(t_track, dtype=np.float64)
    t_dot = np.asarray(t_dot, dtype=np.float64)

    dot_x = np.interp(t_track, t_dot, np.asarray(x_dot, dtype=np.float64))
    dot_y = np.interp(t_track, t_dot, np.asarray(y_dot, dtype=np.float64))

    x_det = _detrend(t_track, np.asarray(x_track, dtype=np.float64), detrend_hz)
    y_det = _detrend(t_track, np.asarray(y_track, dtype=np.float64), detrend_hz)
    dot_x_det = _detrend(t_track, dot_x, detrend_hz)
    dot_y_det = _detrend(t_track, dot_y, detrend_hz)

    n_total = t_track.size
    real_split = bool(held_out and n_total >= 40)
    if real_split:
        idx = np.arange(n_total)
        lo = int(0.4 * n_total)
        hi = int(0.6 * n_total)
        holdout_mask = (idx >= lo) & (idx < hi)
        train_mask = ~holdout_mask
    else:
        holdout_mask = np.ones(n_total, dtype=bool)
        train_mask = holdout_mask

    def _score(sig: np.ndarray, ref: np.ndarray) -> tuple[float, float, int]:
        coef = _affine_fit(sig[train_mask], ref[train_mask])
        pred = _apply_affine(sig, coef)
        r = _pearson_r(pred[holdout_mask], ref[holdout_mask])
        resid = pred[holdout_mask] - ref[holdout_mask]
        finite = np.isfinite(resid)
        rms = float(np.sqrt(np.mean(resid[finite] ** 2))) if finite.any() else float("nan")
        return r, rms, int(finite.sum())

    r_x, rms_x, n_x = _score(x_det, dot_x_det)
    r_y, rms_y, n_y = _score(y_det, dot_y_det)
    return {
        "r_x": r_x,
        "r_y": r_y,
        "rms_x_arcmin": rms_x,
        "rms_y_arcmin": rms_y,
        "n": int(min(n_x, n_y)),
        "held_out": real_split,
        "n_train": int(np.sum(train_mask)),
    }


def speed_percentiles(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    *,
    pcts: tuple[float, ...] = (50, 99, 99.9),
) -> dict:
    """Instantaneous 2-D speed (arcmin/s) percentiles.

    Speed is ``|d(position)| / dt`` between consecutive samples. Reports the
    requested percentiles under keys ``"p{pct}"`` (e.g. ``"p50"``,
    ``"p99.9"``). Useful for surfacing unphysical jumps caused by
    naive/uniform timing assumptions: those jumps are isolated to the far
    upper tail (e.g. p99.9) without moving the median.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x_arcmin, dtype=np.float64)
    y = np.asarray(y_arcmin, dtype=np.float64)
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    m = np.isfinite(dt) & np.isfinite(dx) & np.isfinite(dy) & (dt > 0)
    out: dict = {}
    if not np.any(m):
        for p in pcts:
            out[f"p{p:g}"] = float("nan")
        return out
    speed = np.hypot(dx[m], dy[m]) / dt[m]
    for p in pcts:
        out[f"p{p:g}"] = float(np.percentile(speed, p))
    return out
