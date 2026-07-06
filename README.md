# sdslo-gaze

High-rate (**600–960+ Hz**) two-dimensional gaze tracking and **microsaccade-waveform
recovery** from SD-SLO (spectral-domain scanning laser ophthalmoscope) retinal imaging, with
first-class handling of the scanner **flyback** dead-time.

This is the distilled, production-grade version of an internal research codebase. It keeps only
the validated method, makes the central scientific problem — the no-acquisition flyback gap —
explicit and honest, and ships with tests, config, and reproducible manifests.

## Why this exists

An SD-SLO builds each retinal frame by sweeping a slow galvanometer across 808 columns during an
**active** down-ramp, then **flying back** to the top with **no data acquisition**. The encoded
video hides this: it stores only acquired columns on a uniform clock, so naive timing smears the
unobserved flyback motion into an impossibly short interval and manufactures *supraphysiological*
gaze jumps.

The method here:

1. Tracks gaze **within the active scan** by 2-D strip image registration — the SOTA technique
   (Sheehy et al. 2012: 960 Hz at ≤0.2′; Stevenson et al. 2016: 1920 Hz). Strip width sets the
   rate: `(808 // S) · fps`; `S=8` → **1478 Hz**, well above the 960 Hz operating point.
2. Represents the flyback as **first-class missing data** via a `ScanTiming` / `MeasurementState`
   contract. Every output sample is labeled `observed_active`, `observed_reacq`,
   `predicted_flyback`, `missing_flyback`, or `rejected_mislock`. Accuracy is reported on observed
   samples only; the flyback is a **predict-only** motion-model extrapolation with growing
   uncertainty (a principled replacement for the field's naive zero-order hold — see
   `docs/flyback.md`), never claimed as measured motion.
3. Detects microsaccades and recovers their waveforms where the eye is observed, and reports the
   **duty cycle**: what fraction of the run — and of detected microsaccades — is fully observed
   vs. straddling/inside the blind gap.

The active/flyback split is **not measurable from the provided data** (see `docs/flyback.md`); it
is a hardware-supplied parameter, carried explicitly and bracketed (`active40ms` ↔ `flyback10ms`).

## Install

```bash
pip install -e ".[dev]"          # numpy, scipy, opencv; pytest/matplotlib/pyyaml for dev
```

## Quickstart

```bash
# Point at the source data tree (SLO videos + caches):
export SDSLO_DATA_ROOT=/path/to/gaze-model

# Reproduce the reference result from the cached strip track (fast):
sdslo-gaze validate --which test1 --timing flyback10ms

# Or run this package's own strip tracker on the raw video at ~960 Hz:
sdslo-gaze track --which test1 --source frames --target-hz 960
```

Outputs a JSON + Markdown manifest under `results/` with the measurement-state composition,
precision floor, speed percentiles, microsaccade main sequence, and (if available) held-out
correlation against the pursuit dot.

```python
from sdslo_gaze import pipeline
res = pipeline.run("test1", timing_model="flyback10ms", source="reference")
print(res.strip_hz, res.role_counts, res.microsaccades)
```

## Architecture

See `docs/DESIGN.md`. Modules: `units`, `scan_timing` (the flyback contract), `data_io`,
`strip_tracker`, `dynamics`, `gap_model`, `microsaccade`, `metrics`, `pipeline`, `cli`.

## The flyback problem

See `docs/flyback.md` for the physics, the empirical evidence that the active/flyback split is
unmeasurable from the data, the SOTA literature review, and the handling strategy (including the
hardware roadmap: bidirectional slow-axis or Lissajous scanning to eliminate the gap entirely).

## Data

Raw captures and caches are **not** vendored (large; `.gitignore`d). This package reads them from
`SDSLO_DATA_ROOT`. Canonical testbed: `test1` (808×1000 raster, 14.633 fps, encoded line rate
11,823 Hz, 0.2 Hz pursuit stimulus).

## Status & scope

Validated on the `test1` pursuit raster against the research repo's champion trajectory. Absolute
waveform fidelity against an artificial eye remains future work. Not a medical device.
