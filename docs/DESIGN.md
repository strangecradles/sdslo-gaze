# Architecture

`sdslo-gaze` distills the validated method from the research repo into a small, tested,
reproducible package. It tracks 2-D gaze from SD-SLO retinal video at **600–960+ Hz** and
recovers microsaccade waveforms, treating the scanner flyback as first-class missing data.

## Principles
- **One responsibility per module**, explicit data contracts, no hidden global state.
- **Honesty about the flyback gap**: observed vs predicted vs missing samples are labeled;
  accuracy is reported on observed samples only. See `docs/flyback.md`.
- **Reproducible**: deterministic given inputs; results carry a manifest (params + hashes).
- **No sklearn**; numpy/scipy/opencv/torch(optional). Frozen constants measured from data.

## Module map (`src/sdslo_gaze/`)
| Module | Responsibility | Key API |
|--------|----------------|---------|
| `units.py` | Calibration constants + px↔arcmin↔deg conversions (anisotropic axes) | `col_px_to_arcmin`, `row_px_to_arcmin`, `Units` |
| `scan_timing.py` | **The flyback contract**: `ScanTiming` timing model + `MeasurementState` | `ScanTiming.from_model`, `column_time_s`, `classify` |
| `data_io.py` | Load SLO frames (memmap), stimulus dot, pupil tracker, Cam Right; capture registry | `load_frames`, `load_stimulus`, `load_tracker`, `CAPTURES` |
| `strip_tracker.py` | **SOTA instrument**: 2-D NCC strip registration → high-rate gaze | `track`, `StripTrack`, `select_strip_width` |
| `dynamics.py` | IMM motion prior (pursuit OU + main-sequence saccade) for predict-only | `predict`, `MotionState` |
| `gap_model.py` | Flyback retiming + predict-only propagation + state labeling | `apply_scan_timing`, `bridge_gaps`, `GapTrack` |
| `microsaccade.py` | Detection (EK velocity), main-sequence, waveform + duty-cycle report | `detect`, `main_sequence`, `duty_cycle_report` |
| `metrics.py` | Precision floor (split-half), corr vs dot, speed percentiles | `precision_floor`, `correlate_dot`, `speed_percentiles` |
| `pipeline.py` | End-to-end orchestration → labeled trajectory + microsaccades + manifest | `run` |
| `cli.py` | `sdslo-gaze track|gaps|microsaccades|validate` | — |

## Data contracts
- **`StripTrack`** (strip_tracker → gap_model): arrays `t[s], x_arcmin, y_arcmin, q, contrast,
  infov, frame, strip` + scalars `S, fps, strip_hz, W, H`. x=horizontal(slow/col),
  y=vertical(fast/row). Positions are relative (drift-prone), calibrated to arcmin.
- **`GapTrack`** (gap_model → microsaccade/metrics): `StripTrack` fields re-timed under a
  `ScanTiming`, plus `state: MeasurementState[]` and `observed: bool[]`. Adds predicted
  flyback rows. Accuracy consumers filter on `state == observed_*`.
- **`ScanTiming`**: `fps, sweeps_per_frame(=808), active_ms, flyback_ms` with
  `active_ms + flyback_ms == 1000/fps`. Constructed from a named model (`active40ms`,
  `flyback10ms`) or explicit hardware values.

## Reference validation
The package reproduces the research repo's champion within tolerance by consuming its cached
outputs (`cache/rtrack_test1_S8_incremental_all_p64.npz`, `cache/khz2d_m4_dpf_11823.npz`) and
the pursuit stimulus, then re-deriving metrics through the clean pipeline. See
`scripts/validate_against_reference.py` and `tests/test_pipeline.py`.

## Rate → strip width (test1: W=808, fps=14.633)
`strip_hz = (808 // S) * fps`: S=13→952 Hz, S=8→1478 Hz, S=5→2358 Hz. Default S targets
≥960 Hz. Sheehy 2012 operated 32-px strips → 960 Hz at 0.2′; this repo matches that band.
