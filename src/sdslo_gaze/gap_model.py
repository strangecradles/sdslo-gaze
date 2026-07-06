"""Flyback retiming and predict-only gap bridging: the measurement-state layer.

This module turns a raw strip track (see ``strip_tracker.StripTrack``) into a
:class:`GapTrack`: every sample is re-timed to the true active-scan clock and
labeled with a :class:`~sdslo_gaze.scan_timing.MeasurementState`, and the
scanner flyback (the no-acquisition retrace between frames, see
``docs/flyback.md``) is filled with predict-only propagation or left as
explicit missing data -- never as measured motion.

To keep this module decoupled from the strip-tracker dataclass, the public
functions accept plain arrays (the ``StripTrack`` fields), not the dataclass
itself.

Pipeline (:func:`apply_scan_timing`):
  1. Re-time every sample with :meth:`ScanTiming.strip_time_s`.
  2. Label each sample ``observed_active`` / ``observed_reacq`` per column, or
     mark it ``unlocked`` on low quality / out-of-FOV (no reliable measurement).
  3. Run the motion-plausibility gate on observed samples, rejecting
     physically-implausible jumps as ``rejected_mislock``.
  4. Insert rows across each inter-frame flyback: ``predicted_flyback`` when
     bracketed by finite observations and the gap is short enough to bridge,
     else ``missing_flyback``.

``unlocked`` (low NCC / out-of-FOV, i.e. simply no reliable measurement) is
kept distinct from ``rejected_mislock`` (an *observed* sample discarded by the
motion gate as a physically-implausible jump) so the manifest does not overstate
genuine mislocks. ``observed_mask`` excludes both, so accuracy metrics are
unaffected -- the distinction is about honest reporting, not about which samples
count as measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import MotionState, estimate_velocity, predict_gaussian
from .scan_timing import MeasurementState, ScanTiming, observed_mask
from .units import SWEEPS_PER_FRAME, TEST1_FPS

#: How many trailing observed samples of a frame seed the predict-only velocity.
_VELOCITY_TAIL_K = 5


@dataclass
class GapTrack:
    """A re-timed, gap-labeled gaze track (output of :func:`apply_scan_timing`).

    All arrays are 1-D and share length; sorted by ``t`` ascending. ``state``
    is an ``int8`` array of :class:`~sdslo_gaze.scan_timing.MeasurementState`
    values. Inserted flyback rows carry ``strip = -1`` and ``q = nan`` (no
    real measurement); their ``frame`` is the frame that precedes the gap.
    """

    t: np.ndarray
    x_arcmin: np.ndarray
    y_arcmin: np.ndarray
    state: np.ndarray
    q: np.ndarray
    frame: np.ndarray
    strip: np.ndarray
    timing: ScanTiming

    def observed(self) -> np.ndarray:
        """Boolean mask of samples carrying a real measurement."""
        return observed_mask(self.state)

    def role_counts(self) -> dict[str, int]:
        """Sample count per :class:`MeasurementState`, keyed by ``state.label``."""
        s = np.asarray(self.state, dtype=np.int8)
        return {member.label: int(np.sum(s == int(member))) for member in MeasurementState}

    def duty_fractions(self) -> dict[str, float]:
        """Fraction of rows per role (sums to 1.0, or all-zero if empty)."""
        counts = self.role_counts()
        total = sum(counts.values())
        if total == 0:
            return {k: 0.0 for k in counts}
        return {k: v / total for k, v in counts.items()}


def _reject_implausible_motion(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    q: np.ndarray,
    observed: np.ndarray,
    *,
    max_speed_arcmin_s: float,
    max_micro_step_arcmin: float,
    max_iterations: int = 12,
) -> np.ndarray:
    """Iteratively flag observed samples with physically implausible motion.

    Returns a boolean mask (same length as inputs) of samples to reject: for
    each bad consecutive pair (speed too high, or too big a jump over too
    short an interval), the lower-quality member of the pair is dropped.
    """
    plausible = observed & np.isfinite(x_arcmin) & np.isfinite(y_arcmin)
    rejected = np.zeros(t.shape, dtype=bool)
    for _ in range(max(1, int(max_iterations))):
        idx = np.flatnonzero(plausible)
        if idx.size < 2:
            break
        dt = np.diff(t[idx])
        dist = np.hypot(np.diff(x_arcmin[idx]), np.diff(y_arcmin[idx]))
        speed = np.divide(dist, dt, out=np.full_like(dist, np.inf), where=dt > 0)
        bad_pair = (
            np.isfinite(speed)
            & (dt > 0)
            & (
                (speed > float(max_speed_arcmin_s))
                | ((dt <= 0.0015) & (dist > float(max_micro_step_arcmin)))
            )
        )
        if not np.any(bad_pair):
            break
        to_reject: set[int] = set()
        for k in np.flatnonzero(bad_pair):
            a = int(idx[k])
            b = int(idx[k + 1])
            qa = q[a] if np.isfinite(q[a]) else -np.inf
            qb = q[b] if np.isfinite(q[b]) else -np.inf
            to_reject.add(a if qa <= qb else b)
        if not to_reject:
            break
        rr = np.array(sorted(to_reject), dtype=np.int64)
        plausible[rr] = False
        rejected[rr] = True
    return rejected


def _bridge_flyback_rows(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    frame: np.ndarray,
    observed: np.ndarray,
    S: int,
    timing: ScanTiming,
    *,
    max_bridge_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the inserted flyback rows between each pair of consecutive frames.

    Returns arrays ``(t, x, y, state, q, frame, strip)`` for the new rows only
    (empty arrays if no frame boundary has a bridgeable gap).
    """
    strip_dt = timing.active_dt_s * max(1, int(S))
    frames_present = set(int(f) for f in np.unique(frame))

    new_t: list[float] = []
    new_x: list[float] = []
    new_y: list[float] = []
    new_state: list[int] = []
    new_q: list[float] = []
    new_frame: list[int] = []
    new_strip: list[int] = []

    for f in sorted(frames_present):
        if (f + 1) not in frames_present:
            continue  # no bracketing data after this frame; nothing to bridge
        start, end = timing.flyback_interval_s(f)
        duration = end - start
        if duration <= 0:
            continue
        n_steps = max(1, int(round(duration / max(strip_dt, 1e-12))))
        bridge_t = start + (np.arange(n_steps) + 0.5) * (duration / n_steps)

        idx_f = np.flatnonzero((frame == f) & observed)
        idx_f1 = np.flatnonzero((frame == f + 1) & observed)
        bridgeable = (
            idx_f.size > 0
            and idx_f1.size > 0
            and np.isfinite(x_arcmin[idx_f[-1]])
            and np.isfinite(y_arcmin[idx_f[-1]])
            and np.isfinite(x_arcmin[idx_f1[0]])
            and np.isfinite(y_arcmin[idx_f1[0]])
            and (duration * 1000.0) <= float(max_bridge_ms)
        )

        if bridgeable:
            tail = idx_f[-min(_VELOCITY_TAIL_K, idx_f.size):]
            vel = estimate_velocity(t[tail], x_arcmin[tail], y_arcmin[tail])
            seed = MotionState.from_observations(
                pos=np.array([x_arcmin[idx_f[-1]], y_arcmin[idx_f[-1]]]),
                vel=vel,
                pos_var=1.0,
            )
            last_t = float(t[idx_f[-1]])
            for tb in bridge_t:
                pred = predict_gaussian(seed, float(tb) - last_t)
                new_t.append(float(tb))
                new_x.append(float(pred.pos[0]))
                new_y.append(float(pred.pos[1]))
                new_state.append(int(MeasurementState.PREDICTED_FLYBACK))
                new_q.append(float("nan"))
                new_frame.append(f)
                new_strip.append(-1)
        else:
            for tb in bridge_t:
                new_t.append(float(tb))
                new_x.append(float("nan"))
                new_y.append(float("nan"))
                new_state.append(int(MeasurementState.MISSING_FLYBACK))
                new_q.append(float("nan"))
                new_frame.append(f)
                new_strip.append(-1)

    return (
        np.array(new_t, dtype=np.float64),
        np.array(new_x, dtype=np.float64),
        np.array(new_y, dtype=np.float64),
        np.array(new_state, dtype=np.int8),
        np.array(new_q, dtype=np.float64),
        np.array(new_frame, dtype=np.int64),
        np.array(new_strip, dtype=np.int64),
    )


def apply_scan_timing(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    q: np.ndarray,
    frame: np.ndarray,
    strip: np.ndarray,
    S: int,
    timing: ScanTiming,
    *,
    max_bridge_ms: float = 35.0,
    max_speed_arcmin_s: float = 60000.0,
    max_micro_step_arcmin: float = 8.0,
    ncc_floor: float = 0.35,
    infov: np.ndarray | None = None,
) -> GapTrack:
    """Re-time a strip track to the active-scan clock and label the flyback gap.

    ``t`` is accepted for API symmetry with ``StripTrack`` but is not used for
    timing: every sample is re-timed from scratch via
    ``timing.strip_time_s(frame, strip, S)``, since the encoded per-frame
    clock (what ``t`` typically reflects) smears the flyback into an
    impossibly short interval (see ``docs/flyback.md``).

    See the module docstring for the 4-step pipeline. ``S`` is the strip
    width in columns (``strip * S`` is a strip's starting column).
    """
    x_arcmin = np.asarray(x_arcmin, dtype=np.float64)
    y_arcmin = np.asarray(y_arcmin, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.int64)
    strip = np.asarray(strip, dtype=np.int64)
    n = frame.shape[0]
    if not (x_arcmin.shape[0] == y_arcmin.shape[0] == q.shape[0] == strip.shape[0] == n):
        raise ValueError("t, x_arcmin, y_arcmin, q, frame, strip must all be the same length")

    # (a) re-time every sample to the true active-scan clock.
    t_new = np.asarray(timing.strip_time_s(frame, strip, S), dtype=np.float64)

    # Work in chronological order regardless of input order; the final
    # concatenation with bridge rows is re-sorted again below anyway.
    order = np.argsort(t_new, kind="stable")
    t_new = t_new[order]
    x_arcmin = x_arcmin[order]
    y_arcmin = y_arcmin[order]
    q = q[order]
    frame = frame[order]
    strip = strip[order]
    infov_sorted = np.asarray(infov, dtype=bool)[order] if infov is not None else None

    # (b) label observed samples. Low-quality / out-of-FOV samples are UNLOCKED (no reliable
    # measurement), kept distinct from REJECTED_MISLOCK (a physically-implausible jump flagged
    # by the motion gate in step (c)) so the manifest does not overstate genuine mislocks.
    start_col = strip * int(S)
    state = np.where(
        start_col < timing.reacq_cols,
        int(MeasurementState.OBSERVED_REACQ),
        int(MeasurementState.OBSERVED_ACTIVE),
    ).astype(np.int8)
    locked = q >= float(ncc_floor)
    if infov_sorted is not None:
        locked &= infov_sorted
    state[~locked] = int(MeasurementState.UNLOCKED)

    # (c) motion-plausibility gate on the currently-observed samples.
    observed_now = observed_mask(state)
    rejected = _reject_implausible_motion(
        t_new,
        x_arcmin,
        y_arcmin,
        q,
        observed_now,
        max_speed_arcmin_s=max_speed_arcmin_s,
        max_micro_step_arcmin=max_micro_step_arcmin,
    )
    state[rejected] = int(MeasurementState.REJECTED_MISLOCK)

    # (d) insert flyback rows between consecutive frames.
    observed_final = observed_mask(state)
    (b_t, b_x, b_y, b_state, b_q, b_frame, b_strip) = _bridge_flyback_rows(
        t_new,
        x_arcmin,
        y_arcmin,
        frame,
        observed_final,
        S,
        timing,
        max_bridge_ms=max_bridge_ms,
    )

    all_t = np.concatenate([t_new, b_t])
    all_x = np.concatenate([x_arcmin, b_x])
    all_y = np.concatenate([y_arcmin, b_y])
    all_state = np.concatenate([state, b_state]).astype(np.int8)
    all_q = np.concatenate([q, b_q])
    all_frame = np.concatenate([frame, b_frame])
    all_strip = np.concatenate([strip, b_strip])

    final_order = np.argsort(all_t, kind="stable")
    return GapTrack(
        t=all_t[final_order],
        x_arcmin=all_x[final_order],
        y_arcmin=all_y[final_order],
        state=all_state[final_order],
        q=all_q[final_order],
        frame=all_frame[final_order],
        strip=all_strip[final_order],
        timing=timing,
    )


def sensitivity(
    t: np.ndarray,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    q: np.ndarray,
    frame: np.ndarray,
    strip: np.ndarray,
    S: int,
    models: tuple[str, ...] = ("active40ms", "flyback10ms"),
    *,
    fps: float = TEST1_FPS,
    sweeps_per_frame: int = SWEEPS_PER_FRAME,
    reacq_cols: int = 25,
    max_bridge_ms: float = 35.0,
    max_speed_arcmin_s: float = 60000.0,
    max_micro_step_arcmin: float = 8.0,
    ncc_floor: float = 0.35,
    infov: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Run :func:`apply_scan_timing` under each named timing model.

    Returns ``{model_name: duty_fractions}``, showing how the (unmeasured,
    hardware-supplied) active/flyback split assumption changes the observed
    vs. predicted vs. missing role fractions -- a sensitivity bracket, not a
    single "true" answer (see ``docs/flyback.md``).
    """
    out: dict[str, dict[str, float]] = {}
    for name in models:
        timing = ScanTiming.from_model(
            name, fps=fps, sweeps_per_frame=sweeps_per_frame, reacq_cols=reacq_cols
        )
        track = apply_scan_timing(
            t,
            x_arcmin,
            y_arcmin,
            q,
            frame,
            strip,
            S,
            timing,
            max_bridge_ms=max_bridge_ms,
            max_speed_arcmin_s=max_speed_arcmin_s,
            max_micro_step_arcmin=max_micro_step_arcmin,
            ncc_floor=ncc_floor,
            infov=infov,
        )
        out[name] = track.duty_fractions()
    return out
