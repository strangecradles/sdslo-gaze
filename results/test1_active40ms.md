# sdslo-gaze run — test1 (active40ms timing)

- Strip rate: **1477.9 Hz** (S=8, target 960 Hz)
- Active line rate: 20200.0 Hz; duty cycle 0.585 (active 40.00 ms + flyback 28.34 ms)
- Observed samples: 67,712

## Measurement-state composition
- observed_active: 65,567
- observed_reacq: 2,145
- predicted_flyback: 73,656
- missing_flyback: 0
- rejected_mislock: 7,409
- unlocked: 28,303

## Precision (registration-noise floor, arcmin)
- horizontal: 3.837'  vertical: 1.438'

## Microsaccades
- events: 262
- main sequence: {'slope': 0.5197367597660565, 'intercept': 2.253758874621575, 'r': 0.8069386119153457, 'n': 262}

## Notes
- The active/flyback split is a hardware assumption, not measured from data (see docs/flyback.md).
- Blind-gap microsaccade waveforms are NOT claimed as recovered (`blind_gap_waveform_claim_supported: false`).

## Correlation vs pursuit dot (held-out)
- r_x=0.844, r_y=0.763
