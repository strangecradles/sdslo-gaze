# sdslo-gaze run — test1 (flyback10ms timing)

- Strip rate: **1477.9 Hz** (S=8, target 960 Hz)
- Active line rate: 13850.2 Hz; duty cycle 0.854 (active 58.34 ms + flyback 10.00 ms)
- Observed samples: 68,364

## Measurement-state composition
- observed_active: 66,202
- observed_reacq: 2,162
- predicted_flyback: 17,391
- missing_flyback: 0
- rejected_mislock: 6,757
- unlocked: 28,303

## Precision (registration-noise floor, arcmin)
- horizontal: 2.154'  vertical: 0.808'

## Microsaccades
- events: 494
- main sequence: {'slope': 0.5846323821350499, 'intercept': 2.2696730900235833, 'r': 0.864261300867903, 'n': 494}

## Notes
- The active/flyback split is a hardware assumption, not measured from data (see docs/flyback.md).
- Blind-gap microsaccade waveforms are NOT claimed as recovered (`blind_gap_waveform_claim_supported: false`).

## Correlation vs pursuit dot (held-out)
- r_x=0.871, r_y=0.859
