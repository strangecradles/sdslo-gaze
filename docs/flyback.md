# The SLO Flyback Gap — Physics, Evidence, and Handling

This document explains the central scientific problem this repository addresses and the
design decisions that follow from it. It is the distilled conclusion of an empirical probe
of the source data plus a SOTA literature review.

## 1. The physics

An SD-SLO builds each image frame with two scanners:

| Axis | Scanner | Role | Rate |
|------|---------|------|------|
| Fast (within-line) | resonant | **vertical** gaze; one column = one ~85 µs sweep | ~11.8 kHz line rate |
| Slow (frame) | galvanometer, **sawtooth** ramp | **horizontal** gaze; steps across 808 columns top→bottom, then flies back | 14.633 frames/s |

The slow galvo runs a sawtooth: an **active down-ramp** during which the 808 columns are
acquired, followed by a **flyback** that returns the mirror to the top with **no data
acquisition**. The encoded video (808×1000 @ 14.633 fps) contains *only* the acquired
samples, so the flyback appears as pure elapsed dead-time *between* frames.

- Frame period: **68.34 ms** (1 / 14.633 Hz).
- The active/flyback split is **not encoded in the video** and, as shown below, is **not
  recoverable from any provided data**. It must be supplied from hardware.
- Two bracketing models are carried throughout this repo as *explicit assumptions*:
  - `active40ms`  → 40.0 ms active + 28.3 ms flyback → active line rate 20,200 Hz
  - `flyback10ms` → 58.3 ms active + 10.0 ms flyback → active line rate 13,850 Hz

**Why "unphysical jumps" appear:** if the 808 columns are (wrongly) assumed to
spread uniformly over the whole 68.34 ms period, then the eye displacement accumulated
during the *unobserved* flyback gets attributed to the tiny inter-frame sample interval,
implying eye speeds up to ~438,000 arcmin/s — physically impossible. Correct handling
requires labeling the flyback as missing time, not compressing real motion into it.

## 2. Empirical evidence: the split is unmeasurable from the provided data

A dedicated probe checked every candidate signal:

- **`*_phase.csv`** are per-column *spatial* desinusoiding (dewarp) parameters for a
  different capture geometry (the 600×1200 atlas / zoom captures), not a temporal trace of
  the raster scan. Smooth, no ramp-then-snapback. Dead end.
- **Slow-axis (column) intensity profile** of `test1` is a smooth vignette (min ~55 → peak
  76.85 @ col 584 → ~55), with **no discontinuity/blanking at either frame edge**. This is a
  credible *negative* result: the same method *did* detect a turnaround artifact on the
  orthogonal fast axis (row 0 = 82.4 vs local floor ~43, consistent across frames), so it
  would have caught a slow-axis settling transient if one existed.
- **Per-frame mean intensity** has exactly one sample per 68.34 ms — zero sub-frame
  resolution by construction; its only periodicity (~0.2 Hz) is the pursuit stimulus.
- **`data.py`** contains no flyback/blank/trim logic; it treats all 808 columns as
  uniformly timed (encoded line rate 11,823.5 Hz).

**Verdict:** `active_ms` / `flyback_ms` is a **hardware-supplied config parameter**, not a
measurement. Downstream figures must label any specific split as an assumption and report
sensitivity across the 40/28 ↔ 58/10 bracket.

### Cross-modal channels (measured)
| Channel | Rate | Period | Use |
|---------|------|--------|-----|
| SD-SLO frame | 14.633 Hz | 68.34 ms | primary scan image |
| Cam Right (pupil cam) | **56.876 Hz** | 17.58 ms | VIO-style absolute anchor across the gap |
| Machine pupil tracker | 32.51 Hz | 30.76 ms | coarse gaze reference (already used by gap bridge) |

Neither cross-modal channel observes the scanner, so neither can *measure* the flyback; but
Cam Right (17.6 ms period < flyback ≤ 28 ms) can *bound gaze displacement across* a gap.

## 3. What the literature does (and does not) solve

- **Strip registration is SOTA** for high-rate SLO eye tracking:
  Sheehy et al. 2012 (BOE): 32-px strips × 32/frame → **960 Hz**, residual ≤0.2′.
  Stevenson et al. 2016 (Vis Res): 64 strips → **1920 Hz**, 0.25–0.5′.
  Sheehy et al. 2015 (BOE): 960 Hz active tracking, **0.06′** residual, 2.5 ms latency.
- **Microsaccade waveform recovery needs ≥500–960 Hz** (not just the ~200 Hz detection
  floor): 6–30 ms movements need several samples on the rising flank to recover peak
  velocity and shape (Zuber & Stark 1965 main sequence). 600–960 Hz is the validated band.
- **The flyback gap is essentially unsolved in the SLO literature** — it is either ignored
  (strip-rate papers report only the active-line rate) or handled by a naive **zero-order
  hold** on the last position (Sheehy 2015, active-mirror loop). No principled predict-only
  Bayesian/particle propagation through the gap has SLO precedent.
- **Hardware ways to remove flyback** (roadmap, not usable on the provided raster data):
  bidirectional/triangular slow-axis scanning (OCT-proven, ~100% duty; costs turnaround
  distortion + forward/backward co-registration), or **Lissajous** 2D-resonant scanning
  (Bartuzel et al. 2020: 1.24 kHz, 0.039′, microsaccades to 0.028°, no retrace).

## 4. Handling strategy adopted by this repo (software-only, uses the provided data)

Because we cannot change the hardware or measure the split, the achievable SOTA on the
provided data is to be **honest and maximally informative** about the gap:

1. **Strip tracker at the SOTA operating point.** Non-overlapping strips of width `S` give
   `(808 // S) · fps` Hz. `S=8` → 1478 Hz (above Sheehy's 960 Hz); `S` is configurable to
   land anywhere in the 600–960 Hz validated band. This is the primary microsaccade
   instrument — 2-D template registration tracks *through* microsaccades and is not
   blur-limited (unlike the 1-D particle filter).
2. **First-class `MeasurementState` / `ScanTiming` contract.** Every output sample carries a
   state: `observed_active`, `observed_reacq`, `predicted_flyback` (predict-only, no
   measurement), `missing_flyback` (gap too long to bridge), `rejected_mislock`. Accuracy
   metrics are computed on observed samples only; predicted/missing samples are never
   claimed as measured motion.
3. **Predict-only propagation through the gap** (the literature-novel part). Across the
   flyback, the motion model (pursuit OU + main-sequence saccade prior) propagates state
   with *growing uncertainty* and no measurement update — replacing the field's zero-order
   hold with a velocity-aware Bayesian prediction, explicitly labeled as prediction.
4. **Gap-straddling microsaccade policy.** A microsaccade wholly inside the flyback is
   **not waveform-recoverable** from SLO evidence (stated in the research contract). Its
   *amplitude* is recoverable as the displacement between the last pre-gap and first
   post-gap observed positions (reacquisition), which we report separately from
   fully-observed waveforms.
5. **Duty-cycle reporting.** For a given timing assumption we report the fraction of the run
   that is observed vs predicted vs missing, and the fraction of detected microsaccades that
   fall entirely within active scan (fully recoverable) vs straddle/inside the gap.
6. **Optional cross-modal anchor.** Cam Right (56.9 Hz) can be fused VIO-style as a
   low-rate absolute anchor to constrain drift and bound across-gap displacement.

## 5. Roadmap (hardware, to truly close the gap)
- Feed real `active_ms`/`flyback_ms` from scanner telemetry to collapse the timing bracket.
- Bidirectional slow-axis scanning to reclaim the flyback for acquisition (~+20–40% duty).
- Or Lissajous acquisition to eliminate the raster retrace entirely.

### 5.1 Sinusoidal, no-flyback slow axis — **implemented (software), pending data**
A resonant/sinusoidal slow-axis MEMS mirror removes the flyback outright: both half-periods of
`theta(t) = A·sin(2πf t)` image the field, so the acquisition duty cycle is ~100% and **there is
no no-acquisition gap** — the entire predict-only/missing-data apparatus above collapses to
all-observed. The software to consume such a capture is built and tested:

- `desinusoid.py` resamples each time-uniform sweep onto a uniform **spatial** grid (the samples
  are uniform in time but non-uniform in space — dense at the slow turnarounds, sparse at
  centre). Forward and backward sweeps are placed on the same grid, which co-registers the two
  scan directions for free. Anti-aliased area resampling (default) or cubic interpolation; the
  oversampled turnaround edges can be trimmed (`trim_frac`). Refs: Yang et al. 2015; Giacomelli
  2023.
- `scan_timing.SinusoidTiming` is the gap-free timing contract (zero flyback, ~100% duty) with
  the correct arccos within-sweep time map, so velocity and microsaccade timing stay honest even
  though desinusoided columns are uniform in space, not time.
- `pipeline.run_sinusoid` runs the whole chain (desinusoid → *unchanged* strip tracker → gap-free
  re-timing → microsaccades/metrics). On synthetic bidirectional captures it recovers injected
  motion with no predicted/missing rows (see `tests/test_sinusoid.py`).

The one remaining step is a loader that adapts the real MEMS capture format into the
`(n_sweeps, H, M)` sweep stack `run_sinusoid` expects (plus per-sweep direction and the mirror
frequency / phase). No change to the tracker or the metrics is needed — the trajectory this
package produces is instantly available for eye tracking on that data.

## References
Sheehy 2012 (PMC3469984); Sheehy 2015 (PMC4505698); Stevenson 2016 (PMC4530105);
Ferguson 2010 (PMC3071649); Bartuzel 2020 (PMC7316009); Bedggood & Metha 2017
(PMC5378343); Luo 2021 (PMC8447858); Angelopoulos 2022 (arXiv:2004.03577);
Zuber & Stark 1965 (Science 150:1459); OCT bidirectional (PMC5946793); SLO chapter NBK554043.
