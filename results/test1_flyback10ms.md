# sdslo-gaze run — test1 (flyback10ms timing)

- Strip rate: **1477.9 Hz** (S=8, target 960 Hz)
- Active line rate: 13850.2 Hz; duty cycle 0.854 (active 58.34 ms + flyback 10.00 ms)
- Observed samples: 68,205

## Measurement-state composition
- observed_active: 66,058
- observed_reacq: 2,147
- predicted_flyback: 17,391
- missing_flyback: 0
- rejected_mislock: 6,916
- unlocked: 28,303

## Precision (registration-noise floor, arcmin)
- horizontal: 2.283'  vertical: 0.795'

## Microsaccades
- events: 471
- rate: 10.21 /observed-s (physiological fixation ~1-3/s)
- main sequence: {'slope': 0.706652680070582, 'intercept': 2.1351438984661084, 'r': 0.9147532049638756, 'n': 471}
- surrogate-null control: {'n_shuffle': 20, 'mean_events_per_shuffle': 287.4, 'null_fraction_of_real': 0.6101910828025477} (mean events per shuffled trace; a real population should collapse toward 0)

## Notes
- The active/flyback split is a hardware assumption, not measured from data (see docs/flyback.md).
- Blind-gap microsaccade waveforms are NOT claimed as recovered (`blind_gap_waveform_claim_supported: false`).

## Correlation vs pursuit dot (held-out)
- r_x=0.873, r_y=0.857
