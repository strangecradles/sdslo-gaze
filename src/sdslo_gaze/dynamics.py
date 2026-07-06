"""Oculomotor motion prior, distilled to what the flyback predict-only step needs.

The research repo carries a full IMM particle prior (pursuit Ornstein-Uhlenbeck + main-sequence
saccade). For predict-only propagation across the no-acquisition flyback gap we need two things:

  1. a principled way to extrapolate mean position with *growing uncertainty* when no
     measurement is available (`predict_gaussian`), and
  2. the saccadic main-sequence relation, used by the microsaccade module to sanity-check
     detected events (`main_sequence_peak_velocity`).

Units are arcmin and arcmin/s throughout (converted once from the repo's row/deg constants).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import units

# --- pursuit (fixational drift + smooth pursuit) OU parameters, in arcmin ---
#: Velocity relaxation time constant of the pursuit process.
TAU_PURSUIT_S: float = 0.15
#: Steady-state pursuit velocity std (repo: 6 rows/s -> arcmin/s).
SIGMA_V_PURSUIT_ARCMIN_S: float = 6.0 * units.ARCMIN_PER_ROW_PX  # ~2.89 '/s
#: Hard per-step acceleration cap (repo: 4000 rows/s^2 -> arcmin/s^2).
ACCEL_CAP_ARCMIN_S2: float = 4000.0 * units.ARCMIN_PER_ROW_PX

# --- saccadic main sequence (amplitude -> peak velocity), in arcmin ---
#: Main-sequence knee amplitude.
A0_ARCMIN: float = 6.5
#: Saturating peak velocity (repo: VMAX 103000 rows/s -> arcmin/s ~ 826 deg/s).
VMAX_ARCMIN_S: float = 103000.0 * units.ARCMIN_PER_ROW_PX  # ~49582 '/s


def main_sequence_peak_velocity(amp_arcmin: np.ndarray | float) -> np.ndarray | float:
    """Peak velocity (arcmin/s) predicted by the saccadic main sequence for an amplitude.

    ``v_peak(A) = VMAX * (1 - exp(-A / A0))`` -- the saturating form used to generate and to
    validate (micro)saccades. Monotonic; ~linear for small microsaccades, saturating for large.
    """
    a = np.asarray(amp_arcmin, dtype=np.float64)
    return VMAX_ARCMIN_S * (1.0 - np.exp(-a / A0_ARCMIN))


@dataclass(frozen=True)
class MotionState:
    """Gaussian belief about gaze kinematics at one instant (arcmin, arcmin/s).

    ``pos``/``vel`` are 2-vectors [x=horizontal, y=vertical]. ``pos_var`` is the position
    variance per axis (arcmin^2); ``vel`` is treated as a point estimate whose contribution to
    position uncertainty grows with the propagation interval.
    """

    pos: np.ndarray            # (2,) arcmin
    vel: np.ndarray            # (2,) arcmin/s
    pos_var: np.ndarray        # (2,) arcmin^2

    @staticmethod
    def from_observations(
        pos: np.ndarray, vel: np.ndarray, pos_var: float | np.ndarray = 1.0
    ) -> "MotionState":
        pos = np.asarray(pos, dtype=np.float64).reshape(2)
        vel = np.asarray(vel, dtype=np.float64).reshape(2)
        pv = np.asarray(pos_var, dtype=np.float64)
        if pv.ndim == 0:
            pv = np.full(2, float(pv))
        return MotionState(pos=pos, vel=vel, pos_var=pv.reshape(2))


def predict_gaussian(state: MotionState, dt: float) -> MotionState:
    """Predict-only propagation across ``dt`` seconds with NO measurement update.

    Velocity relaxes toward zero under the pursuit OU (mean reverting), position integrates the
    (decaying) velocity, and position variance grows from the pursuit velocity diffusion. This
    replaces the field's naive zero-order hold with a velocity-aware, uncertainty-growing
    extrapolation -- the samples it produces are labeled ``PREDICTED_FLYBACK`` and are never
    counted as measurements.
    """
    if dt < 0:
        raise ValueError("dt must be >= 0")
    tau = TAU_PURSUIT_S
    decay = float(np.exp(-dt / tau))
    # Mean velocity decays toward 0 (OU); mean displacement is the integral of v(t).
    disp = state.vel * tau * (1.0 - decay)
    new_pos = state.pos + disp
    new_vel = state.vel * decay
    # Position variance growth: diffusion of the pursuit velocity integrated over dt.
    # Var[integral of OU velocity] ~ sigma_v^2 * tau * dt for dt on the order of tau (a standard,
    # slightly conservative bound); exact enough for the ~10-28 ms flyback.
    var_growth = (SIGMA_V_PURSUIT_ARCMIN_S ** 2) * tau * dt
    new_var = state.pos_var + var_growth
    return replace(state, pos=new_pos, vel=new_vel, pos_var=new_var)


def estimate_velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    """Robust local velocity (arcmin/s) at the last sample from the last ``k`` finite samples.

    Used to seed :func:`predict_gaussian` at the start of a flyback gap. Returns a 2-vector.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return np.zeros(2)
    idx = idx[-min(k, idx.size):]
    tt = t[idx]
    if tt[-1] - tt[0] <= 0:
        return np.zeros(2)
    # least-squares slope per axis
    A = np.vstack([tt - tt.mean(), np.ones_like(tt)]).T
    vx = np.linalg.lstsq(A, x[idx], rcond=None)[0][0]
    vy = np.linalg.lstsq(A, y[idx], rcond=None)[0][0]
    return np.array([vx, vy], dtype=np.float64)
