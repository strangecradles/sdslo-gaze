"""Data loading: SD-SLO frames, stimulus, pupil tracker, and reference caches.

The raw captures and derived caches live in the source research repo (they are large and are
never vendored into this clean repo). Point this module at that tree via the ``SDSLO_DATA_ROOT``
environment variable or the ``data_root`` argument; the default is the sibling ``gaze-model``
checkout.

Only lightweight, well-typed reductions are exposed. Frame pixels are memory-mapped, never fully
read into RAM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default location of the source data tree (override with SDSLO_DATA_ROOT or data_root=).
_DEFAULT_DATA_ROOT = Path(
    os.environ.get("SDSLO_DATA_ROOT", str(Path.home() / "Kodiak" / "gaze-model"))
)


@dataclass(frozen=True)
class Capture:
    """A registered SD-SLO capture."""

    which: str
    slo_stem: str          # video/csv stem under calibration/
    kind: str              # "raster" | "xscan"
    stimulus: str          # "pursuit" | "calib_grid"


#: Canonical captures (mirrors the research repo's data.CAPTURES).
CAPTURES: dict[str, Capture] = {
    "test1": Capture("test1", "video_playback_test1_20260605_100051", "raster", "pursuit"),
    "test2": Capture("test2", "video_playback_test2_20260605_100257", "xscan", "pursuit"),
    "Athton1": Capture("Athton1", "video_playback_Athton1_20260605_095610", "xscan", "calib_grid"),
}


# --- frames ---
@dataclass
class Frames:
    """Memory-mapped SD-SLO frame stack (n, rows=1000, cols=808), float32, plus fps."""

    array: np.ndarray  # np.memmap (n, H, W)
    fps: float
    which: str

    @property
    def n(self) -> int:
        return int(self.array.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.array.shape)  # type: ignore[return-value]


def load_frames(which: str = "test1", *, data_root: str | os.PathLike | None = None,
                max_frames: int | None = None) -> Frames:
    """Load SD-SLO frames.

    Prefers the research repo's decoded memmap ``cache/frames_<which>_all.dat`` + ``.npz``
    sidecar (fast, no decode). Falls back to decoding the mp4 with OpenCV if the cache is
    absent.
    """
    root = _resolve_root(data_root)
    npz = root / "cache" / f"frames_{which}_all.npz"
    dat = root / "cache" / f"frames_{which}_all.dat"
    if npz.exists() and dat.exists():
        # Plain-array caches produced by our own research repo; allow_pickle not required.
        meta = np.load(npz, allow_pickle=False)
        shape = tuple(int(v) for v in meta["shape"])  # (n, H, W)
        fps = float(meta["fps"])
        arr = np.memmap(dat, dtype=np.float32, mode="r", shape=shape)
        if max_frames is not None:
            arr = arr[:max_frames]
        return Frames(array=arr, fps=fps, which=which)
    return _decode_mp4(which, root, max_frames)


def _resolve_root(dr: str | os.PathLike | None) -> Path:
    root = Path(dr) if dr is not None else _DEFAULT_DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"data root {root} not found; set SDSLO_DATA_ROOT or pass data_root=..."
        )
    return root


def _decode_mp4(which: str, root: Path, max_frames: int | None) -> Frames:
    import cv2

    cap = CAPTURES[which]
    path = root / "calibration" / f"{cap.slo_stem}_SLO_0.mp4"
    if not path.exists():
        raise FileNotFoundError(f"SLO video not found: {path}")
    vc = cv2.VideoCapture(str(path))
    fps = float(vc.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ok, fr = vc.read()
        if not ok:
            break
        if fr.ndim == 3:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(fr.astype(np.float32))
    vc.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return Frames(array=np.stack(frames), fps=fps, which=which)


def deband(frame: np.ndarray) -> np.ndarray:
    """Remove per-row then per-column mean (de-vignette / de-band), as in the source repo."""
    f = frame.astype(np.float32)
    f = f - f.mean(axis=1, keepdims=True)
    f = f - f.mean(axis=0, keepdims=True)
    return f


# --- stimulus (pursuit dot) ---
@dataclass
class Stimulus:
    t: np.ndarray        # s
    x_norm: np.ndarray   # [-1, 1] horizontal
    y_norm: np.ndarray   # [-1, 1] vertical


def load_stimulus(name: str = "pursuit", *, data_root: str | os.PathLike | None = None) -> Stimulus:
    """Load the stimulus dot trajectory (normalized screen coords)."""
    root = _resolve_root(data_root)
    path = root / "stimuli" / f"{name}.csv"
    rows = np.genfromtxt(path, delimiter=",", names=True)
    return Stimulus(
        t=np.asarray(rows["t_sec"], dtype=np.float64),
        x_norm=np.asarray(rows["x_norm"], dtype=np.float64),
        y_norm=np.asarray(rows["y_norm"], dtype=np.float64),
    )


# --- machine pupil tracker + Cam Right rate ---
@dataclass
class MachineTrack:
    t: np.ndarray        # s (playback_time_s)
    x: np.ndarray        # right_x (raw tracker units)
    y: np.ndarray        # right_y
    rate_hz: float


def load_tracker(which: str = "test1", *, eye: str = "right",
                 data_root: str | os.PathLike | None = None) -> MachineTrack:
    """Load the ~32 Hz machine pupil tracker (right eye by default)."""
    root = _resolve_root(data_root)
    cap = CAPTURES[which]
    path = root / "calibration" / f"{cap.slo_stem}.csv"
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    t = np.asarray(rows["playback_time_s"], dtype=np.float64)
    x = np.asarray(rows[f"{eye}_x"], dtype=np.float64)
    y = np.asarray(rows[f"{eye}_y"], dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    dt = np.median(np.diff(t)) if t.size > 1 else np.nan
    rate = float(1.0 / dt) if dt and np.isfinite(dt) and dt > 0 else float("nan")
    return MachineTrack(t=t, x=x, y=y, rate_hz=rate)


# --- reference caches (for validation against the research repo) ---
def load_reference_strip_track(which: str = "test1", S: int = 8, ref_mode: str = "incremental",
                               span: str = "all", pad: int = 64,
                               data_root: str | os.PathLike | None = None) -> dict:
    """Load a cached ``raster_track`` output from the research repo (arrays as a dict)."""
    root = _resolve_root(data_root)
    path = root / "cache" / f"rtrack_{which}_S{S}_{ref_mode}_{span}_p{pad}.npz"
    if not path.exists():
        raise FileNotFoundError(f"reference strip-track cache not found: {path}")
    # Plain-array cache from our own research repo; pickle not required.
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_reference_pf(method: str = "m4_dpf_11823",
                      data_root: str | os.PathLike | None = None) -> dict:
    """Load a cached kHz particle-filter trajectory from the research repo."""
    root = _resolve_root(data_root)
    path = root / "cache" / f"khz2d_{method}.npz"
    if not path.exists():
        raise FileNotFoundError(f"reference PF cache not found: {path}")
    # Plain-array cache from our own research repo; pickle not required.
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_champion_csv(method: str = "m4_dpf_11823",
                      data_root: str | os.PathLike | None = None) -> dict[str, np.ndarray]:
    """Load the research repo's champion gap-aware CSV as columns of arrays.

    Carries dot ground truth already in arcmin (``dot_x_arcmin``/``dot_y_arcmin``) aligned to the
    reconstruction -- the cleanest ground-truth reference for validation.
    """
    root = _resolve_root(data_root)
    path = root / "results" / f"champion_gap_aware_{method}.csv"
    if not path.exists():
        raise FileNotFoundError(f"champion CSV not found: {path}")
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {name: np.asarray(rows[name]) for name in rows.dtype.names}
