"""Oculomotor motion prior, distilled to what the flyback predict-only step needs.

The previous repo carries a full IMM particle prior (pursuit Ornstein-Uhlenbeck + main-sequence
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

# --- pursuit / fixational-drift parameters, in arcmin ---
#: Velocity relaxation time constant of the pursuit process.
TAU_PURSUIT_S: float = 0.15
#: Fixational-drift diffusion coefficient (arcmin^2/s). Drift is a random walk in position,
#: so position variance grows ~2*D*t (Cherici et al. 2012 measured D ~ 5-20 across observers;
#: Rucci models use ~40). 20 is a mid-range value; it sets how fast the predict-only flyback
#: uncertainty grows across the no-acquisition gap.
DRIFT_DIFFUSION_ARCMIN2_S: float = 20.0

# --- saccadic main sequence (amplitude -> peak velocity), in arcmin ---
#: Saturating peak velocity (VMAX 103000 rows/s -> arcmin/s ~ 826 deg/s; Bahill et al. 1975).
VMAX_ARCMIN_S: float = 103000.0 * units.ARCMIN_PER_ROW_PX  # ~49582 '/s
#: Main-sequence knee amplitude, set so the small-amplitude slope VMAX/A0 ~ 47 /s reproduces
#: the Zuber & Stark (1965) main sequence (peak velocity [deg/s] ~ 47 * amplitude [deg]).
#: A 12' microsaccade then peaks near ~9 deg/s, as observed; the old 6.5' knee was ~160x too
#: small and put a 12' microsaccade at ~700 deg/s.
A0_ARCMIN: float = VMAX_ARCMIN_S / 47.0  # ~1055 arcmin (~17.6 deg)


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
    # Position variance growth: fixational drift is a random walk in position, so its variance
    # grows ~2*D*dt (Cherici et al. 2012). This is the honest uncertainty band on a predicted
    # flyback sample -- ~0.9' after a 10 ms gap, ~1.5' after 28 ms -- and it is what
    # gap_model surfaces as `sigma_arcmin` on each PREDICTED_FLYBACK row.
    var_growth = 2.0 * DRIFT_DIFFUSION_ARCMIN2_S * dt
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
