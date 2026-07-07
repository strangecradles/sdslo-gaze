"""The scanner-timing / measurement-state contract.

This is the conceptual centerpiece of the package. An SD-SLO acquires 808 columns during an
*active* slow-axis down-ramp, then *flies back* to the top with no acquisition. The encoded
video hides this: it stores only acquired columns at a uniform frame clock, so naive timing
smears the unobserved flyback motion into an impossibly short interval.

We make the timing explicit and label every output sample with a `MeasurementState`, so that
downstream accuracy claims are computed on *observed* samples only and the flyback interval is
represented as prediction-with-growing-uncertainty or as missing data -- never as measured
motion. See ``docs/flyback.md``.

The active/flyback split is NOT measurable from the provided data (see the empirical probe in
``docs/flyback.md``); it is a hardware-supplied parameter. Two bracketing models are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .units import SWEEPS_PER_FRAME, TEST1_FPS


class MeasurementState(IntEnum):
    """Provenance of a single gaze sample.

    ``observed`` states carry real image measurements and are the only ones used for accuracy
    metrics. ``predicted_flyback`` is a motion-model extrapolation across the no-acquisition
    gap (labeled, never claimed as measured). ``missing_flyback`` marks a gap too long to
    bridge. ``rejected_mislock`` marks an observed sample discarded as a physically implausible
    jump (wrong-feature lock).
    """

    OBSERVED_ACTIVE = 0       #: measured during the active scan
    OBSERVED_REACQ = 1        #: measured, first columns after a frame boundary (re-acquisition)
    PREDICTED_FLYBACK = 2     #: predict-only extrapolation across the flyback (no measurement)
    MISSING_FLYBACK = 3       #: flyback gap left unfilled (too long / no bracketing lock)
    REJECTED_MISLOCK = 4      #: observed but rejected as implausible motion (wrong-feature lock)
    UNLOCKED = 5              #: low NCC quality / out-of-FOV; no reliable measurement (not a mislock)

    @property
    def is_observed(self) -> bool:
        return self in _OBSERVED

    @property
    def label(self) -> str:
        return self.name.lower()


_OBSERVED = frozenset({MeasurementState.OBSERVED_ACTIVE, MeasurementState.OBSERVED_REACQ})


def observed_mask(states: np.ndarray) -> np.ndarray:
    """Boolean mask of samples that carry a real measurement (for accuracy metrics)."""
    s = np.asarray(states, dtype=np.int8)
    return (s == MeasurementState.OBSERVED_ACTIVE) | (s == MeasurementState.OBSERVED_REACQ)


@dataclass(frozen=True)
class ScanTiming:
    """Explicit active-scan / flyback timing model for one capture.

    ``active_ms + flyback_ms`` must equal the frame period ``1000 / fps``. Construct via
    :meth:`from_model` (named bracket) or :meth:`from_hardware` (real telemetry).
    """

    fps: float = TEST1_FPS
    sweeps_per_frame: int = SWEEPS_PER_FRAME
    active_ms: float = 58.339
    flyback_ms: float = 10.0
    #: Number of leading columns after a frame boundary tagged as re-acquisition transient.
    reacq_cols: int = 25
    #: Human-readable name of the timing assumption (for manifests / figures).
    label: str = "flyback10ms"

    def __post_init__(self) -> None:
        period = 1000.0 / self.fps
        if abs((self.active_ms + self.flyback_ms) - period) > 1e-3:
            raise ValueError(
                f"active_ms ({self.active_ms}) + flyback_ms ({self.flyback_ms}) = "
                f"{self.active_ms + self.flyback_ms:.4f} ms must equal the frame period "
                f"{period:.4f} ms (fps={self.fps})."
            )
        if self.active_ms <= 0 or self.flyback_ms < 0:
            raise ValueError("active_ms must be > 0 and flyback_ms must be >= 0.")

    # --- derived timing ---
    @property
    def frame_period_s(self) -> float:
        return 1.0 / self.fps

    @property
    def active_s(self) -> float:
        return self.active_ms / 1000.0

    @property
    def flyback_s(self) -> float:
        return self.flyback_ms / 1000.0

    @property
    def active_dt_s(self) -> float:
        """Real time between consecutive acquired columns (active window / sweeps)."""
        return self.active_s / self.sweeps_per_frame

    @property
    def active_line_hz(self) -> float:
        """True line rate during the active scan (higher than the encoded rate)."""
        return self.sweeps_per_frame / self.active_s

    @property
    def duty_cycle(self) -> float:
        """Fraction of each frame period spent acquiring (active / total)."""
        return self.active_s / self.frame_period_s

    # --- timestamps ---
    def column_time_s(
        self, frame: np.ndarray | int, col: np.ndarray | int
    ) -> np.ndarray | float:
        """Acquisition time of column ``col`` in ``frame`` (columns span only the active window)."""
        frame = np.asarray(frame, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        return frame * self.frame_period_s + (col + 0.5) * self.active_dt_s

    def strip_time_s(
        self, frame: np.ndarray | int, strip: np.ndarray | int, strip_width: int
    ) -> np.ndarray | float:
        """Acquisition time of a strip (its column midpoint) under this timing."""
        frame = np.asarray(frame, dtype=np.float64)
        strip = np.asarray(strip, dtype=np.float64)
        center_col = strip * strip_width + strip_width / 2.0
        return frame * self.frame_period_s + center_col * self.active_dt_s

    def flyback_interval_s(self, frame: int) -> tuple[float, float]:
        """(start, end) real time of the flyback gap immediately after ``frame``."""
        start = frame * self.frame_period_s + self.active_s
        end = (frame + 1) * self.frame_period_s
        return start, end

    def is_reacq_col(self, col: np.ndarray | int) -> np.ndarray | bool:
        """Whether a column falls in the leading re-acquisition transient of its frame."""
        return np.asarray(col) < self.reacq_cols

    # --- constructors ---
    @classmethod
    def from_model(
        cls, name: str, fps: float = TEST1_FPS, sweeps_per_frame: int = SWEEPS_PER_FRAME,
        reacq_cols: int = 25,
    ) -> "ScanTiming":
        """Build from a named bracketing assumption. See :data:`TIMING_MODELS`."""
        if name not in TIMING_MODELS:
            raise KeyError(f"unknown timing model {name!r}; choose from {sorted(TIMING_MODELS)}")
        spec = TIMING_MODELS[name]
        period_ms = 1000.0 / fps
        if "active_ms" in spec:
            active_ms = float(spec["active_ms"])
            flyback_ms = period_ms - active_ms
        else:
            flyback_ms = float(spec["flyback_ms"])
            active_ms = period_ms - flyback_ms
        return cls(fps=fps, sweeps_per_frame=sweeps_per_frame, active_ms=active_ms,
                   flyback_ms=flyback_ms, reacq_cols=reacq_cols, label=name)

    @classmethod
    def from_hardware(
        cls, *, active_ms: float | None = None, flyback_ms: float | None = None,
        fps: float = TEST1_FPS, sweeps_per_frame: int = SWEEPS_PER_FRAME, reacq_cols: int = 25,
    ) -> "ScanTiming":
        """Build from measured scanner telemetry: supply exactly one of active/flyback ms."""
        if (active_ms is None) == (flyback_ms is None):
            raise ValueError("supply exactly one of active_ms or flyback_ms")
        period_ms = 1000.0 / fps
        if active_ms is None:
            active_ms = period_ms - float(flyback_ms)
        else:
            flyback_ms = period_ms - float(active_ms)
        return cls(fps=fps, sweeps_per_frame=sweeps_per_frame, active_ms=float(active_ms),
                   flyback_ms=float(flyback_ms), reacq_cols=reacq_cols, label="hardware")

    def summary(self) -> dict:
        return {
            "label": self.label,
            "fps": self.fps,
            "sweeps_per_frame": self.sweeps_per_frame,
            "frame_period_ms": 1000.0 * self.frame_period_s,
            "active_ms": self.active_ms,
            "flyback_ms": self.flyback_ms,
            "active_line_hz": self.active_line_hz,
            "duty_cycle": self.duty_cycle,
            "reacq_cols": self.reacq_cols,
            "note": "active/flyback split is a hardware assumption, not measured from data",
        }


#: Named bracketing assumptions for the unmeasured active/flyback split (see docs/flyback.md).
TIMING_MODELS: dict[str, dict] = {
    "active40ms": {"active_ms": 40.0},   # 40 active + 28.3 flyback -> active line rate 20200 Hz
    "flyback10ms": {"flyback_ms": 10.0}, # 58.3 active + 10 flyback -> active line rate 13850 Hz
    "uniform": {"flyback_ms": 0.0},      # no flyback: encoded uniform clock (baseline, wrong)
}


@dataclass(frozen=True)
class SinusoidTiming:
    """Timing contract for a **sinusoidal, no-flyback** slow axis (resonant MEMS mirror).

    Where :class:`ScanTiming` models a sawtooth galvo (active ramp + blind flyback), this models
    a slow axis driven as ``theta(t) = A*sin(2*pi*f*t)``. Each usable half-period is a monotonic
    edge-to-edge *sweep*; both half-periods image the field, so there is **no flyback gap** --
    duty cycle is ~100% (minus any trimmed turnarounds) and no sample is ever ``predicted`` or
    ``missing``. After desinusoiding (see :mod:`sdslo_gaze.desinusoid`) a sweep's columns are
    uniform in *space*; their acquisition *time* follows the arccos map, which
    :meth:`strip_time_s` applies so velocities/microsaccade timing stay honest.

    This class is duck-compatible with the subset of the :class:`ScanTiming` surface that
    :func:`sdslo_gaze.gap_model.apply_scan_timing` uses (``strip_time_s``, ``reacq_cols``,
    ``flyback_interval_s`` -> zero-length, ``active_dt_s``, ``is_reacq_col``, ``summary``), so the
    same re-timing / labelling / mislock-gating pipeline runs on a sinusoidal capture with the
    flyback-bridging step reducing to a no-op.

    Each desinusoided sweep is one "frame" (index ``0, 1, 2, ...``); by default even frames are
    forward sweeps and odd frames backward (bidirectional acquisition).
    """

    f_scan_hz: float                    #: mirror drive frequency; one sweep = half a period
    cols_per_sweep: int = SWEEPS_PER_FRAME  #: desinusoided spatial columns kept per sweep (N)
    trim_frac: float = 0.0              #: fraction of the field trimmed at each turnaround edge
    reacq_cols: int = 0                 #: no reacquisition transient without a flyback
    label: str = "sinusoid"

    def __post_init__(self) -> None:
        if self.f_scan_hz <= 0:
            raise ValueError("f_scan_hz must be > 0")
        if self.cols_per_sweep < 1:
            raise ValueError("cols_per_sweep must be >= 1")
        if not (0.0 <= self.trim_frac < 0.5):
            raise ValueError("trim_frac must be in [0, 0.5)")

    # --- derived timing ---
    @property
    def sweep_period_s(self) -> float:
        """Duration of one sweep (half the mirror period)."""
        return 1.0 / (2.0 * self.f_scan_hz)

    @property
    def sweep_rate_hz(self) -> float:
        """Sweeps acquired per second (both half-periods used)."""
        return 2.0 * self.f_scan_hz

    @property
    def frame_period_s(self) -> float:
        return self.sweep_period_s

    @property
    def duty_cycle(self) -> float:
        """Fraction of wall-clock time spent acquiring (~1.0 minus trimmed turnarounds)."""
        return 1.0 - 2.0 * self.trim_frac

    @property
    def active_dt_s(self) -> float:
        """Nominal mean time between spatial columns (sweep duration / columns)."""
        return self.sweep_period_s / self.cols_per_sweep

    def _time_frac(self, frame: np.ndarray, center_col: np.ndarray) -> np.ndarray:
        """Fractional time within a sweep of a strip centred at ``center_col`` (spatial column)."""
        span = 1.0 - 2.0 * self.trim_frac
        u = self.trim_frac + center_col / self.cols_per_sweep * span
        tf = np.arccos(np.clip(1.0 - 2.0 * u, -1.0, 1.0)) / np.pi   # forward
        backward = (frame.astype(np.int64) % 2) == 1
        return np.where(backward, 1.0 - tf, tf)

    def strip_time_s(
        self, frame: np.ndarray | int, strip: np.ndarray | int, strip_width: int
    ) -> np.ndarray | float:
        """Acquisition time of a strip, honouring the sinusoidal within-sweep time map."""
        frame = np.asarray(frame, dtype=np.float64)
        strip = np.asarray(strip, dtype=np.float64)
        center_col = strip * strip_width + strip_width / 2.0
        tf = self._time_frac(frame, center_col)
        return frame * self.sweep_period_s + tf * self.sweep_period_s

    def flyback_interval_s(self, frame: int) -> tuple[float, float]:
        """No flyback: a zero-length gap (so gap bridging is a no-op)."""
        start = (frame + 1) * self.sweep_period_s
        return start, start

    def is_reacq_col(self, col: np.ndarray | int) -> np.ndarray | bool:
        return np.asarray(col) < self.reacq_cols

    def summary(self) -> dict:
        return {
            "label": self.label,
            "scan_mode": "sinusoid",
            "f_scan_hz": self.f_scan_hz,
            "sweep_rate_hz": self.sweep_rate_hz,
            "sweep_period_ms": 1000.0 * self.sweep_period_s,
            "cols_per_sweep": self.cols_per_sweep,
            "trim_frac": self.trim_frac,
            "duty_cycle": self.duty_cycle,
            "flyback_ms": 0.0,
            "reacq_cols": self.reacq_cols,
            "note": "sinusoidal slow axis: no flyback gap, ~100% duty, bidirectional sweeps",
        }
