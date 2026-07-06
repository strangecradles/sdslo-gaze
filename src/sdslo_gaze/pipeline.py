"""End-to-end pipeline: SD-SLO video -> labeled high-rate gaze -> microsaccades + metrics.

Orchestrates the modules into one reproducible run and emits a manifest. The trajectory can be
produced by this package's own strip tracker (``source="frames"``) or loaded from the research
repo's cached strip-track output (``source="reference"``) for fast validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import data_io, dynamics, gap_model, metrics, microsaccade, units
from .scan_timing import ScanTiming, observed_mask
from .strip_tracker import StripTrack, track


@dataclass
class RunResult:
    which: str
    timing: dict
    target_hz: float
    strip_width: int
    strip_hz: float
    n_observed: int
    role_counts: dict
    duty: dict
    precision_arcmin: dict
    speed_arcmin_s: dict
    microsaccades: dict
    dot_correlation: dict | None = None
    manifest_path: str | None = None
    _gap: object = field(default=None, repr=False)


def _strip_track_from_reference(which: str, S: int, data_root) -> StripTrack:
    """Build a StripTrack from the research repo's cached raster_track output."""
    d = data_io.load_reference_strip_track(which, S=S, data_root=data_root)
    along = np.asarray(d["along_px"], dtype=np.float64)  # horizontal (slow/col)
    perp = np.asarray(d["perp_px"], dtype=np.float64)     # vertical (fast/row)
    return StripTrack(
        t=np.asarray(d["t"], dtype=np.float64),
        x_arcmin=np.asarray(units.col_px_to_arcmin(along)),
        y_arcmin=np.asarray(units.row_px_to_arcmin(perp)),
        x_px=along,
        y_px=perp,
        q=np.asarray(d["q"], dtype=np.float64),
        contrast=np.asarray(d["contrast"], dtype=np.float64),
        infov=np.asarray(d["infov"], dtype=bool),
        frame=np.asarray(d["frame"], dtype=np.int32),
        strip=np.asarray(d["strip"], dtype=np.int32),
        S=int(d["S"]),
        fps=float(d["fps"]),
        strip_hz=float(d["strip_hz"]),
        W=int(d["W"]),
        H=int(d["H"]),
    )


def run(
    which: str = "test1",
    *,
    timing_model: str = "flyback10ms",
    target_hz: float = 960.0,
    source: str = "reference",
    S: int | None = None,
    max_frames: int | None = None,
    data_root: str | None = None,
    out_dir: str | None = "results",
    with_dot: bool = True,
) -> RunResult:
    """Run the full pipeline for one capture and (optionally) write a manifest.

    Parameters
    ----------
    source : "reference" loads the cached strip track (fast, for validation); "frames" runs this
        package's own strip tracker on the SD-SLO video.
    timing_model : named active/flyback assumption (see scan_timing.TIMING_MODELS).
    with_dot : if True and the champion CSV is available, report held-out correlation vs the dot.
    """
    # 1. strip track (this package's tracker, or the reference cache)
    if source == "reference":
        st = _strip_track_from_reference(which, S=S or 8, data_root=data_root)
    elif source == "frames":
        frames = data_io.load_frames(which, data_root=data_root, max_frames=max_frames)
        st = track(frames, S=S, target_hz=target_hz)
    else:
        raise ValueError("source must be 'reference' or 'frames'")

    # 2. explicit scanner timing (the flyback contract)
    timing = ScanTiming.from_model(timing_model, fps=st.fps, reacq_cols=25)

    # 3. re-time + label measurement states + predict-only flyback bridging
    gap = gap_model.apply_scan_timing(
        st.t, st.x_arcmin, st.y_arcmin, st.q, st.frame, st.strip, st.S, timing,
        infov=st.infov,
    )
    obs = gap.observed()

    # 4. microsaccades on observed samples only
    events = microsaccade.detect(gap.t, gap.x_arcmin, gap.y_arcmin, observed=obs)
    ms_main = microsaccade.main_sequence(events)
    # State array is at the strip cadence, not the encoded line rate -> pass the true rate so
    # the events/observed-second estimate is correct.
    duty = microsaccade.duty_cycle_report(gap.state, events, sample_hz=st.strip_hz)

    # 5. metrics (observed-only)
    prec = metrics.precision_floor(gap.t[obs], gap.x_arcmin[obs], gap.y_arcmin[obs])
    speed = metrics.speed_percentiles(gap.t[obs], gap.x_arcmin[obs], gap.y_arcmin[obs])

    dot_corr = None
    if with_dot:
        try:
            champ = data_io.load_champion_csv(data_root=data_root)
            dt = np.asarray(champ["t_s"], dtype=np.float64)
            dx = np.asarray(champ["dot_x_arcmin"], dtype=np.float64)
            dy = np.asarray(champ["dot_y_arcmin"], dtype=np.float64)
            dfin = np.isfinite(dt) & np.isfinite(dx) & np.isfinite(dy)
            dot_corr = metrics.correlate_dot(
                gap.t[obs], gap.x_arcmin[obs], gap.y_arcmin[obs],
                dt[dfin], dx[dfin], dy[dfin],
            )
        except (FileNotFoundError, KeyError, ValueError):
            dot_corr = None

    result = RunResult(
        which=which,
        timing=timing.summary(),
        target_hz=target_hz,
        strip_width=st.S,
        strip_hz=st.strip_hz,
        n_observed=int(obs.sum()),
        role_counts=gap.role_counts(),
        duty=duty,
        precision_arcmin=prec,
        speed_arcmin_s=speed,
        microsaccades={
            "n_events": len(events),
            "main_sequence": ms_main,
        },
        dot_correlation=dot_corr,
        _gap=gap,
    )

    if out_dir is not None:
        result.manifest_path = _write_manifest(result, out_dir, timing_model)
    return result


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_manifest(result: RunResult, out_dir: str, timing_model: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{result.which}_{timing_model}"
    payload = {
        "which": result.which,
        "package": "sdslo-gaze",
        "timing": result.timing,
        "target_hz": result.target_hz,
        "strip_width": result.strip_width,
        "strip_hz": result.strip_hz,
        "n_observed": result.n_observed,
        "role_counts": result.role_counts,
        "duty_cycle_report": result.duty,
        "precision_arcmin": result.precision_arcmin,
        "speed_arcmin_s": result.speed_arcmin_s,
        "microsaccades": result.microsaccades,
        "dot_correlation": result.dot_correlation,
        "blind_gap_waveform_claim_supported": False,
    }
    jpath = out / f"{stem}.json"
    jpath.write_text(json.dumps(_to_jsonable(payload), indent=2))
    (out / f"{stem}.md").write_text(_render_md(payload))
    return str(jpath)


def _render_md(p: dict) -> str:
    t = p["timing"]
    lines = [
        f"# sdslo-gaze run — {p['which']} ({t['label']} timing)",
        "",
        f"- Strip rate: **{p['strip_hz']:.1f} Hz** (S={p['strip_width']}, target {p['target_hz']:.0f} Hz)",
        f"- Active line rate: {t['active_line_hz']:.1f} Hz; duty cycle {t['duty_cycle']:.3f} "
        f"(active {t['active_ms']:.2f} ms + flyback {t['flyback_ms']:.2f} ms)",
        f"- Observed samples: {p['n_observed']:,}",
        "",
        "## Measurement-state composition",
    ]
    for k, v in p["role_counts"].items():
        lines.append(f"- {k}: {v:,}")
    prec = p["precision_arcmin"]
    lines += [
        "",
        "## Precision (registration-noise floor, arcmin)",
        f"- horizontal: {prec.get('rms_x', float('nan')):.3f}'  vertical: {prec.get('rms_y', float('nan')):.3f}'",
        "",
        "## Microsaccades",
        f"- events: {p['microsaccades']['n_events']}",
        f"- main sequence: {p['microsaccades']['main_sequence']}",
        "",
        "## Notes",
        "- The active/flyback split is a hardware assumption, not measured from data (see docs/flyback.md).",
        "- Blind-gap microsaccade waveforms are NOT claimed as recovered "
        "(`blind_gap_waveform_claim_supported: false`).",
    ]
    if p.get("dot_correlation"):
        dc = p["dot_correlation"]
        lines += ["", "## Correlation vs pursuit dot (held-out)",
                  f"- r_x={dc.get('r_x', float('nan')):.3f}, r_y={dc.get('r_y', float('nan')):.3f}"]
    return "\n".join(lines) + "\n"
