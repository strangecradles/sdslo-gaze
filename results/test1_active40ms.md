# sdslo-gaze run — test1 (active40ms timing)

- Strip rate: **1477.9 Hz** (S=8, target 960 Hz)
- Active line rate: 20200.0 Hz; duty cycle 0.585 (active 40.00 ms + flyback 28.34 ms)
- Observed samples: 65,059

## Measurement-state composition
- observed_active: 63,040
- observed_reacq: 2,019
- predicted_flyback: 73,656
- missing_flyback: 0
- rejected_mislock: 10,062
- unlocked: 28,303

## Precision (registration-noise floor, arcmin)
- horizontal: 2.125'  vertical: 0.695'

## Microsaccades
- events: 271
- rate: 6.16 /observed-s (physiological fixation ~1-3/s)
- main sequence: {'slope': 0.6023418234041775, 'intercept': 2.0956601293709025, 'r': 0.8625825420228458, 'n': 271}
- surrogate-null control: {'n_shuffle': 20, 'mean_events_per_shuffle': 231.8, 'null_fraction_of_real': 0.8553505535055351} (mean events per shuffled trace; a real population should collapse toward 0)

## Notes
- The active/flyback split is a hardware assumption, not measured from data (see docs/flyback.md).
- Blind-gap microsaccade waveforms are NOT claimed as recovered (`blind_gap_waveform_claim_supported: false`).

## Correlation vs pursuit dot (held-out)
- r_x=0.845, r_y=0.751
