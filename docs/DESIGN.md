# Architecture

`sdslo-gaze` distills the validated method from the research repo into a small, tested,
reproducible package. It tracks 2-D gaze from SD-SLO video at **600–960+ Hz** and
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
| `scan_timing.py` | **The flyback contract**: `ScanTiming` (sawtooth) + `SinusoidTiming` (no-flyback) + `MeasurementState` | `ScanTiming.from_model`, `SinusoidTiming`, `column_time_s` |
| `desinusoid.py` | **Sinusoidal rectification**: resample a time-uniform sinusoidal sweep → uniform spatial grid (fwd/bwd, anti-aliased) | `desinusoid_sweep`, `desinusoid_stack`, `output_time_frac` |
| `data_io.py` | Load SLO frames (memmap), stimulus dot, pupil tracker, Cam Right; capture registry | `load_frames`, `load_stimulus`, `load_tracker`, `CAPTURES` |
| `strip_tracker.py` | **SOTA instrument**: 2-D NCC strip registration → high-rate gaze | `track`, `StripTrack`, `select_strip_width` |
| `dynamics.py` | IMM motion prior (pursuit OU + main-sequence saccade) for predict-only | `predict`, `MotionState` |
| `gap_model.py` | Flyback retiming + predict-only propagation + state labeling | `apply_scan_timing`, `bridge_gaps`, `GapTrack` |
| `microsaccade.py` | Detection (EK velocity), main-sequence, waveform + duty-cycle report | `detect`, `main_sequence`, `duty_cycle_report` |
| `metrics.py` | Precision floor (split-half), corr vs dot, speed percentiles | `precision_floor`, `correlate_dot`, `speed_percentiles` |
| `pipeline.py` | End-to-end orchestration → labeled trajectory + microsaccades + manifest | `run` (sawtooth), `run_sinusoid` (no-flyback) |
| `cli.py` | `sdslo-gaze track|gaps|microsaccades|validate` | — |

## Data contracts
- **`StripTrack`** (strip_tracker → gap_model): arrays `t[s], x_arcmin, y_arcmin, q, contrast,
  infov, frame, strip` + scalars `S, fps, strip_hz, W, H`. x=horizontal(slow/col),
  y=vertical(fast/row). Positions are relative (drift-prone), calibrated to arcmin.
- **`GapTrack`** (gap_model → microsaccade/metrics): `StripTrack` fields re-timed under a
  timing model, plus `state: MeasurementState[]`, `observed: bool[]`, and `sigma_arcmin[]` (the
  predict-only 1σ position uncertainty: growing on predicted-flyback rows, `inf` on missing,
  `nan` on measured). Accuracy consumers filter on `state == observed_*`.
- **`ScanTiming`** (sawtooth): `fps, sweeps_per_frame(=808), active_ms, flyback_ms` with
  `active_ms + flyback_ms == 1000/fps`. Constructed from a named model (`active40ms`,
  `flyback10ms`) or explicit hardware values.
- **`SinusoidTiming`** (no-flyback): `f_scan_hz, cols_per_sweep, trim_frac`. Duck-compatible
  with the `ScanTiming` surface `gap_model` uses, but with a zero-length flyback (so bridging is
  a no-op) and an arccos within-sweep time map. Even frames = forward sweeps, odd = backward.

## Scan modes
- **Sawtooth** (`run`): the galvo raster the repo started from — active ramp + blind flyback.
  The flyback is carried as first-class missing data (predict-only / missing), the central
  honesty problem (see `docs/flyback.md`).
- **Sinusoidal, no-flyback** (`run_sinusoid`): a resonant/sinusoidal slow-axis MEMS mirror.
  Both half-periods image the field, so there is **no flyback gap** — duty ~100%, no
  predicted/missing rows. `desinusoid.py` resamples each time-uniform sweep onto a uniform
  spatial grid (forward/backward co-registered) before the *unchanged* strip tracker runs; the
  gap-free `SinusoidTiming` re-times it. This is the hardware target: once real MEMS sweeps are
  captured, wiring a loader into `run_sinusoid` is the only remaining step.

## Reference validation
The package reproduces the research repo's champion within tolerance by consuming its cached
outputs (`cache/rtrack_test1_S8_incremental_all_p64.npz`, `cache/khz2d_m4_dpf_11823.npz`) and
the pursuit stimulus, then re-deriving metrics through the clean pipeline. See
`scripts/validate_against_reference.py` and `tests/test_pipeline.py`.

## Rate → strip width (test1: W=808, fps=14.633)
`strip_hz = (808 // S) * fps`: S=13→907 Hz, S=12→980 Hz, S=8→1478 Hz, S=5→2356 Hz. The
default `target_hz=960` picks the widest strip that still clears 960 Hz → S=12 (980 Hz);
the reference validation cache is S=8 → 1478 Hz. Sheehy 2012 operated 32-px strips → 960 Hz;
this repo runs in the same rate band.
