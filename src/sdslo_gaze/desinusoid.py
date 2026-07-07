"""desinusoid.py -- spatial rectification of a sinusoidally-scanned slow axis.

A resonant / sinusoidal slow-axis MEMS mirror scans position ``theta(t) = A*sin(2*pi*f*t)``.
Over one usable half-period (a "sweep") the mirror moves monotonically edge-to-edge, so samples
taken on a uniform time clock are **uniform in time but non-uniform in space**: dense near the
turnarounds (where the mirror is slow) and sparse at field centre (where it is fastest). This
module resamples each time-uniform sweep onto a uniform **spatial** grid -- "desinusoiding", the
standard AOSLO/OCT rectification (Yang et al. 2015; Giacomelli 2023).

Two consequences matter for this package:

* **No flyback.** Unlike the sawtooth raster the repo started from, a sinusoidal slow axis images
  the field on *both* half-periods, so the acquisition duty cycle is ~100% and there is no
  no-acquisition gap to predict across. The whole ``predicted_flyback`` / ``missing_flyback``
  machinery collapses to all-observed (see :class:`~sdslo_gaze.scan_timing.SinusoidTiming`).
* **Forward/backward co-registration is free.** A backward sweep images the field in the opposite
  spatial direction; resampling it onto the *same* ascending spatial grid as the forward sweeps
  (this module flips it internally) puts both directions in one coordinate frame, so the existing
  strip tracker consumes the stream unchanged.

The spatial coordinate ``u`` runs 0..1 across the field (0 = one turnaround edge, 1 = the other).
A uniform-time input column ``j`` of a forward sweep of ``M`` columns sits at
``u_in(j) = (1 - cos(pi*(j+0.5)/M)) / 2`` -- the inverse of the sinusoid. The output grid is
uniform in ``u`` over the kept (un-trimmed) central range.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def input_spatial_coord(n_in: int) -> np.ndarray:
    """Spatial coordinate ``u in (0, 1)`` of each uniform-time input column (forward sweep).

    ``u_in(j) = (1 - cos(pi*(j+0.5)/n_in)) / 2`` -- monotonically increasing, dense near the
    edges (``u -> 0, 1``) and sparse at centre (``u = 0.5``).
    """
    if n_in < 1:
        raise ValueError("n_in must be >= 1")
    j = np.arange(n_in, dtype=np.float64)
    return (1.0 - np.cos(np.pi * (j + 0.5) / n_in)) / 2.0


def output_time_frac(n_out: int, direction: str = "forward", *, trim_frac: float = 0.0) -> np.ndarray:
    """Fractional time within the sweep (``[0, 1)``) of each uniform-space output column.

    A forward sweep acquires the low-``u`` edge first, so the time fraction rises with column;
    a backward sweep acquires the high-``u`` edge first, so it is ``1 - forward``. This is the
    per-column timestamp map the timing model uses -- desinusoided columns are uniform in
    *space*, not in *time*.
    """
    _check_trim(trim_frac)
    span = 1.0 - 2.0 * trim_frac
    u = trim_frac + (np.arange(n_out, dtype=np.float64) + 0.5) / n_out * span
    tf = np.arccos(np.clip(1.0 - 2.0 * u, -1.0, 1.0)) / np.pi
    if direction == "backward":
        return 1.0 - tf
    if direction != "forward":
        raise ValueError(f"direction must be 'forward' or 'backward'; got {direction!r}")
    return tf


def desinusoid_sweep(
    sweep: np.ndarray,
    n_out: int,
    direction: str = "forward",
    *,
    trim_frac: float = 0.0,
    antialias: bool = True,
) -> np.ndarray:
    """Resample one time-uniform sweep ``(H, M)`` onto a uniform spatial grid ``(H, n_out)``.

    ``direction`` selects the sinusoid half-period; a ``"backward"`` sweep is flipped internally
    so its output lands on the same ascending spatial grid as forward sweeps (co-registration).
    ``trim_frac`` drops that fraction of the field at *each* turnaround edge (the oversampled,
    distortion-prone extremes); ``trim_frac=0.1`` keeps the central 80%.

    ``antialias=True`` uses exact area (box) resampling: each output pixel is the spatial-overlap
    average of the input columns it covers, which averages the dense edge samples instead of
    point-sampling them (no aliasing). ``antialias=False`` uses cubic interpolation (sharper at
    the sparse centre, but no anti-alias at the edges).
    """
    sweep = np.asarray(sweep, dtype=np.float64)
    if sweep.ndim != 2:
        raise ValueError(f"sweep must be 2-D (H, M); got shape {sweep.shape}")
    if n_out < 1:
        raise ValueError("n_out must be >= 1")
    _check_trim(trim_frac)
    if direction == "backward":
        sweep = sweep[:, ::-1]
    elif direction != "forward":
        raise ValueError(f"direction must be 'forward' or 'backward'; got {direction!r}")

    M = sweep.shape[1]
    u_in = input_spatial_coord(M)                       # (M,) ascending in (0, 1)
    span = 1.0 - 2.0 * trim_frac
    out_edges = trim_frac + np.arange(n_out + 1, dtype=np.float64) / n_out * span

    if antialias:
        return _area_resample(sweep, u_in, out_edges)
    centers = 0.5 * (out_edges[:-1] + out_edges[1:])
    frac_idx = np.interp(centers, u_in, np.arange(M, dtype=np.float64))
    return _cubic_cols(sweep, frac_idx)


def desinusoid_stack(
    sweeps: np.ndarray,
    n_out: int,
    directions: np.ndarray | None = None,
    *,
    trim_frac: float = 0.0,
    antialias: bool = True,
) -> np.ndarray:
    """Desinusoid a stack of sweeps ``(n_sweeps, H, M)`` -> ``(n_sweeps, H, n_out)``.

    ``directions`` is a length-``n_sweeps`` array/sequence of ``"forward"``/``"backward"`` (or
    0/1); if ``None`` it alternates forward, backward, forward, ... (bidirectional resonant scan),
    which is the ~100%-duty no-flyback acquisition this package targets.
    """
    sweeps = np.asarray(sweeps, dtype=np.float64)
    if sweeps.ndim != 3:
        raise ValueError(f"sweeps must be (n_sweeps, H, M); got shape {sweeps.shape}")
    n = sweeps.shape[0]
    dirs = _normalize_directions(directions, n)
    out = np.empty((n, sweeps.shape[1], n_out), dtype=np.float64)
    for i in range(n):
        out[i] = desinusoid_sweep(
            sweeps[i], n_out, dirs[i], trim_frac=trim_frac, antialias=antialias
        )
    return out


# --- internals --------------------------------------------------------------


def _check_trim(trim_frac: float) -> None:
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError(f"trim_frac must be in [0, 0.5); got {trim_frac}")


def _normalize_directions(directions, n: int) -> list[str]:
    if directions is None:
        return ["forward" if i % 2 == 0 else "backward" for i in range(n)]
    dirs = list(directions)
    if len(dirs) != n:
        raise ValueError(f"directions has length {len(dirs)}, expected {n}")
    out = []
    for d in dirs:
        if d in ("forward", "backward"):
            out.append(d)
        elif d in (0, "0"):
            out.append("forward")
        elif d in (1, "1"):
            out.append("backward")
        else:
            raise ValueError(f"bad direction {d!r}")
    return out


def _area_resample(sweep: np.ndarray, u_in: np.ndarray, out_edges: np.ndarray) -> np.ndarray:
    """Exact box/area resampling of each row onto the ``out_edges`` spatial bins.

    Each input column ``j`` owns the spatial interval between the midpoints to its neighbours
    (ends clamped to [0, 1]); an output pixel's value is the length-weighted average of the input
    columns overlapping its bin. Implemented via the piecewise-linear cumulative integral so it is
    exact and vectorised over rows.
    """
    M = sweep.shape[1]
    in_edges = np.empty(M + 1, dtype=np.float64)
    in_edges[1:-1] = 0.5 * (u_in[:-1] + u_in[1:])
    in_edges[0] = 0.0
    in_edges[-1] = 1.0
    widths = np.diff(in_edges)                                   # (M,)
    widths = np.where(widths > 0, widths, 1e-12)

    # Cumulative integral of the piecewise-constant row at the input edges: (H, M+1).
    cumF = np.zeros((sweep.shape[0], M + 1), dtype=np.float64)
    np.cumsum(sweep * widths, axis=1, out=cumF[:, 1:])

    # Evaluate the (piecewise-linear) cumulative integral at each output edge.
    oe = np.clip(out_edges, 0.0, 1.0)
    seg = np.clip(np.searchsorted(in_edges, oe, side="right") - 1, 0, M - 1)
    frac = (oe - in_edges[seg]) / widths[seg]
    F_out = cumF[:, seg] + frac * (sweep[:, seg] * widths[seg])  # (H, n_out+1)

    out_w = np.diff(oe)
    out_w = np.where(out_w > 0, out_w, 1e-12)
    return (F_out[:, 1:] - F_out[:, :-1]) / out_w


def _cubic_cols(sweep: np.ndarray, frac_idx: np.ndarray) -> np.ndarray:
    """Cubic-interpolate each row of ``sweep`` at fractional column indices ``frac_idx``."""
    H = sweep.shape[0]
    rows = np.repeat(np.arange(H, dtype=np.float64), frac_idx.size)
    cols = np.tile(frac_idx, H)
    vals = map_coordinates(sweep, [rows, cols], order=3, mode="nearest")
    return vals.reshape(H, frac_idx.size)
