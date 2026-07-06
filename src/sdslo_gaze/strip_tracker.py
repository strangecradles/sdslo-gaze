"""strip_tracker.py -- 2-D NCC strip registration for high-rate SD-SLO gaze recovery.

The decisive insight for high-rate 2-D gaze from a RASTER SLO capture: each frame is
built column-by-column over the frame period, so a STRIP of ``S`` consecutive
slow-axis (column) samples is a genuine 2-D retinal patch acquired at a
sub-frame instant. Registering each strip against the previous frame with 2-D
normalized cross-correlation (OpenCV ``TM_CCOEFF_NORMED``) recovers gaze at
strip rate ``(W // S) * fps`` -- hundreds of Hz, well above the frame rate --
without the perp aliasing that a 1-D line-scan match would suffer (a 2-D patch
match against a 2-D reference is unique). This is the Sheehy/Roorda TSLO
method (Biomed. Opt. Express 3(10):2611, 2012: 960 Hz @ 0.66') distilled to a
faithful, tested reimplementation.

Axis convention (see :mod:`sdslo_gaze.units`): the SD-SLO frame is
``(rows=FAST=vertical, cols=SLOW=horizontal)``. A strip ``frame[:, c:c+S]`` is
``(H, S)``; its acquisition time is ``t = f/fps + (c + S/2)/W/fps``.

Only the ``incremental`` reference mode is implemented here: each frame's
strips are 2-D-registered against the PREVIOUS frame (small inter-frame
motion -> robust); per-frame motion is the median strip shift across the
frame, and the within-frame residual (strip shift minus the frame median)
rides on top of the cumulative sum. This yields a RELATIVE (drift-prone)
trajectory, calibrated to arcmin via the measured axis scales in
:mod:`sdslo_gaze.units`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import units
from .data_io import Frames, deband

#: Default target strip rate (Hz). Sheehy 2012 operated ~960 Hz strips.
DEFAULT_TARGET_HZ: float = 960.0
#: Default local 2-D search half-window (px) around the zero-motion prediction.
DEFAULT_PAD: int = 64
#: Per-strip NCC peak floor for "in FOV / locked".
Q_FOV: float = 0.35
#: Raw strip contrast floor, as a fraction of the frame-stack median contrast.
CONTRAST_FRAC: float = 0.5


def select_strip_width(
    frame_width: int,
    fps: float,
    target_hz: float = DEFAULT_TARGET_HZ,
    prefer_at_least: bool = True,
) -> int:
    """Choose a non-overlapping strip width for a target sampling rate.

    ``track`` uses complete, non-overlapping strips only, so the realized rate
    is ``(frame_width // S) * fps``. Among strip widths whose realized rate
    meets or exceeds ``target_hz``, returns the one whose rate is closest to
    ``target_hz`` (i.e. the widest, most robust strip that still clears the
    target); if none clears the target, falls back to the strip width whose
    rate is closest to ``target_hz`` overall.
    """
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    candidates = [
        (S, (frame_width // S) * fps)
        for S in range(1, frame_width + 1)
        if frame_width // S > 0
    ]
    pool = [(S, rate) for S, rate in candidates if rate >= target_hz] if prefer_at_least else []
    if not pool:
        pool = candidates
    S, _ = min(pool, key=lambda item: (abs(item[1] - target_hz), item[0]))
    return int(S)


@dataclass
class StripTrack:
    """Per-strip 2-D gaze trajectory recovered by :func:`track`.

    ``x``/``along`` is the horizontal (slow/column) axis; ``y``/``perp`` is
    the vertical (fast/row) axis. Positions are RELATIVE (drift-prone,
    incremental reference mode) but calibrated to arcmin.
    """

    t: np.ndarray          # (N,) s, per-strip acquisition time
    x_arcmin: np.ndarray   # (N,) horizontal gaze (slow/column axis), arcmin, relative
    y_arcmin: np.ndarray   # (N,) vertical gaze (fast/row axis), arcmin, relative
    x_px: np.ndarray       # (N,) raw column-pixel displacement (horizontal)
    y_px: np.ndarray       # (N,) raw row-pixel displacement (vertical)
    q: np.ndarray          # (N,) NCC peak quality
    contrast: np.ndarray   # (N,) raw strip contrast (std, post-deband)
    infov: np.ndarray      # (N,) bool, locked / in-FOV gate
    frame: np.ndarray      # (N,) int32 frame index
    strip: np.ndarray      # (N,) int32 strip index within frame
    S: int                 # cols per strip
    fps: float              # frame rate (Hz)
    strip_hz: float         # (W // S) * fps
    W: int                  # frame width (cols)
    H: int                  # frame height (rows)

    @property
    def n(self) -> int:
        return int(len(self.t))


def _nz(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalize (guards against a flat/zero-variance patch)."""
    s = x.std()
    return ((x - x.mean()) / (s if s > 0 else 1.0)).astype(np.float32)


def _parabolic(c: np.ndarray, k: int) -> float:
    """Sub-pixel peak refine: fit a parabola through the 3 samples around index k."""
    if 0 < k < len(c) - 1:
        a, b, d = float(c[k - 1]), float(c[k]), float(c[k + 1])
        den = a - 2.0 * b + d
        if abs(den) > 1e-12:
            return float(k + 0.5 * (a - d) / den)
    return float(k)


def track(
    frames: Frames | np.ndarray,
    *,
    fps: float | None = None,
    S: int | None = None,
    target_hz: float = DEFAULT_TARGET_HZ,
    pad: int = DEFAULT_PAD,
    ref_mode: str = "incremental",
) -> StripTrack:
    """Strip-track a frame stack at ``S`` cols/strip -> per-strip 2-D gaze.

    ``frames`` may be a :class:`~sdslo_gaze.data_io.Frames` (its ``fps`` is
    used unless overridden) or a plain ``(n, H, W)`` array, in which case
    ``fps`` must be given. Each frame is ``deband``-ed before matching. Each
    strip (full frame height x ``S`` columns) is registered against the SAME
    columns of the previous frame, padded by ``pad`` px, via
    ``cv2.matchTemplate(TM_CCOEFF_NORMED)`` with parabolic sub-pixel peak
    refinement. The per-frame motion is the median strip shift; the
    within-frame residual (strip shift minus that median) rides on the
    cumulative sum of per-frame motion, giving each strip its own timestamped
    2-D displacement.

    Only ``ref_mode="incremental"`` is implemented.
    """
    if ref_mode != "incremental":
        raise ValueError(f"unsupported ref_mode {ref_mode!r}; only 'incremental' is implemented")

    if isinstance(frames, Frames):
        array = frames.array
        if fps is None:
            fps = frames.fps
    else:
        array = np.asarray(frames)
        if fps is None:
            raise ValueError("fps must be given when frames is a plain array")

    if array.ndim != 3:
        raise ValueError(f"frames must be (n, H, W); got shape {array.shape}")
    n, H, W = array.shape
    if S is None:
        S = select_strip_width(W, fps, target_hz)
    nstrip = W // S
    if nstrip < 1:
        raise ValueError(f"strip width S={S} exceeds frame width {W}")
    if n < 2:
        raise ValueError("track requires at least 2 frames")

    Tf = 1.0 / fps

    # Representative median contrast (post-deband) across a sparse frame subsample, for the
    # in-FOV contrast gate.
    sample_frames = range(0, n, max(1, n // 20))
    con_samples = [
        float(deband(array[f])[:, c * S:(c + 1) * S].std())
        for f in sample_frames
        for c in range(nstrip)
    ]
    con_med = float(np.median(con_samples)) if con_samples else 1.0

    t_out: list[float] = []
    x_out: list[float] = []
    y_out: list[float] = []
    q_out: list[float] = []
    con_out: list[float] = []
    frame_out: list[int] = []
    strip_out: list[int] = []

    refp: np.ndarray | None = None
    cumx = cumy = 0.0
    for f in range(n):
        deb = deband(array[f])
        cur = _nz(deb)
        if refp is None:
            refp = cv2.copyMakeBorder(cur, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            continue

        fx = np.zeros(nstrip)
        fy = np.zeros(nstrip)
        fq = np.zeros(nstrip)
        for s in range(nstrip):
            c = s * S
            strip = cur[:, c:c + S]
            reg = refp[:, c:c + S + 2 * pad]
            r = cv2.matchTemplate(reg, strip, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            fy[s] = _parabolic(r[:, loc[0]], loc[1]) - pad
            fx[s] = _parabolic(r[loc[1], :], loc[0]) - pad
            fq[s] = mx

        med_x = float(np.median(fx))
        med_y = float(np.median(fy))
        cumx += med_x
        cumy += med_y

        for s in range(nstrip):
            c = s * S
            t_out.append(f * Tf + (c + S / 2) / W * Tf)
            x_out.append(cumx + (fx[s] - med_x))
            y_out.append(cumy + (fy[s] - med_y))
            q_out.append(fq[s])
            con_out.append(float(deb[:, c:c + S].std()))
            frame_out.append(f)
            strip_out.append(s)

        refp = cv2.copyMakeBorder(cur, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    t = np.asarray(t_out, dtype=np.float64)
    x_px = np.asarray(x_out, dtype=np.float32)
    y_px = np.asarray(y_out, dtype=np.float32)
    q = np.asarray(q_out, dtype=np.float32)
    contrast = np.asarray(con_out, dtype=np.float32)
    frame = np.asarray(frame_out, dtype=np.int32)
    strip = np.asarray(strip_out, dtype=np.int32)
    infov = (q > Q_FOV) & (contrast > CONTRAST_FRAC * con_med)

    return StripTrack(
        t=t,
        x_arcmin=np.asarray(units.col_px_to_arcmin(x_px)),
        y_arcmin=np.asarray(units.row_px_to_arcmin(y_px)),
        x_px=x_px,
        y_px=y_px,
        q=q,
        contrast=contrast,
        infov=infov,
        frame=frame,
        strip=strip,
        S=int(S),
        fps=float(fps),
        strip_hz=float((W // S) * fps),
        W=int(W),
        H=int(H),
    )
