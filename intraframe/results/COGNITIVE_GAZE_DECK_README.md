# Cognitive gaze-tracking evidence deck

## Deliverables

- `cognitive_gaze_tracking_evidence_deck.pptx` — editable 16:9 PowerPoint.
- `cognitive_gaze_tracking_evidence_deck.pdf` — Keynote-rendered review copy.
- `cognitive_gaze_tracking_evidence_deck.tex` — editable 39-frame LaTeX Beamer translation.
- `cognitive_gaze_tracking_evidence_deck_latex.pdf` — compiled Beamer review copy.
- `cognitive_gaze_tracking_explained_deck.tex` — expanded 48-frame Beamer deck that teaches the
  instrument, timing model, registration pipeline, evidence, safety regimes, and architecture.
- `cognitive_gaze_tracking_explained_deck.pdf` — compiled explanatory deck.
- `cognitive_gaze_tracking_evidence_paper.tex` — 25-page paper-style technical treatment.
- `cognitive_gaze_tracking_evidence_paper.pdf` — compiled paper-style review copy.
- `cognitive_gaze_evidence.json` — machine-readable evidence and assumption ledger.
- `cognitive_gaze_figure_manifest.json` — evidence, input, generator, and figure hashes.
- `cognitive_gaze_presenter_notes.json` — slide-indexed explanation sidecar.
- `cognitive_gaze_figures/` — reproducible high-resolution figures used by the deck.
- `cognitive_gaze_rendered/` — rendered slide PNGs and contact sheets used for visual QA.
- `../scripts/build_cognitive_gaze_figures.py` — figure generator.
- `../scripts/build_cognitive_gaze_deck.py` — deck generator.

## Evidence policy

Every quantitative statement is classified as measured, derived, literature, fixture,
program fact, hypothesis, or missing. The deck does not silently convert:

- synthetic known-truth recovery into biological validation;
- predicted dead-time samples into observations;
- nominal eye geometry into subject-specific retinal scale;
- overlapping strip updates into independent photon measurements; or
- a parameterized exposure framework into a laser-safety determination.

The primary engineering source is identified by filename and content hash:

`SLO_Scan_Bandwidth_Sampling_Technical_Note_v2.docx`
SHA-256: `e9fb9728c77d352bcd55ccb6eb66979ee415b1c9d416d32b517f1c924a6c33c4`

The note's 10 kHz fast scan, 10 MHz measured analog bandwidth, 50 MS/s candidate ADC,
10 µm Airy-disk hypothesis, and 24 mm nominal eye model are kept distinct from measurements
and assumptions in the legacy `test1` capture.

## Important corrections

- A 10 kHz full sine provides 20,000 one-way sweeps per second when both directions are used.
- Combining 20 kline/s with 808 slow-axis lines gives 40.4 ms active time. Combining that
  value with the current 14.633 fps frame period leaves 27.939 ms unexplained by active lines.
  This is a conditional derivation, not a measured slow-axis flyback waveform.
- A 24-line strip at 20 kline/s has a 1.20 ms aperture. A 6-line stride gives 0.30 ms output
  spacing (3.33 kHz during active acquisition), but adjacent estimates share 75% of their lines.
- Ocular tremor content is reported around 40–100 Hz. A 200 Hz rate is only the ideal Nyquist
  floor for 100 Hz content; practical waveform and spectral recovery calls for at least about
  500 Hz. Bowers et al. sampled direct retinal motion at 1,920 Hz, enabling analysis through
  the 960 Hz Nyquist ceiling.
- Tremor intrinsically demands arcsecond-scale precision. High-rate drift increments and fine
  direction estimates for the smallest microsaccades can also become arcsecond-limited.
- Arcsecond performance reported for AOSLO is not inherited by this conventional SD-SLO;
  model-eye and biological validation remain required.
- Pupil dynamics are assigned to the eye-camera channel. Literature guidance reports little
  pupil activity above 9 Hz and typical acquisition around 50–60 Hz, with higher rates useful
  for latency and artifact handling.

## Laser-safety boundary

The deck shows a symbolic scanning-exposure workflow and a derived 22.9 ns center crossing time
for the assumed 10 µm diameter at 436.8 m/s. It does **not** calculate an allowed power because
the following production inputs are absent: wavelength/spectrum, corneal power and tolerance,
beam diameter/divergence, actual scan/blanking distribution, exposure duration, optical
transmission, source subtense, pupil/aperture conditions, and scanner-stall exposure.

Applicable standards and regulatory pathways must be selected by the formal safety review.
Laser-product classification under IEC 60825-1/21 CFR, ophthalmic patient exposure under
ISO 15004-2 or FDA-recognized ANSI Z80.36, and FDA medical-device classification are separate
evaluations. The ledger also cites IEC TS 60825-13:2026, Laser Notice 56, product code MYC, and
the FDA scanning-safeguard checklist for 21 CFR 1040.10(f)(9).

## Rebuild

From the repository root:

```bash
PYTHONPATH=intraframe/src python intraframe/scripts/build_cognitive_gaze_figures.py \
  --champion-csv "$HOME/Kodiak/gaze-model/results/champion_gap_aware_m4_dpf_11823.csv"
PYTHONPATH=intraframe/src python intraframe/scripts/build_cognitive_gaze_deck.py
```

The figure build requires the sibling `~/Kodiak/gaze-model` data tree used by the repository's
existing loaders and fails if the pursuit result is unavailable, preventing stale-figure reuse.
The figure build writes a hash manifest only after every required input is validated and every
figure succeeds. The deck refuses figures whose hashes do not match the current ledger and manifest.
The PDF and rendered-slide PNGs
are review artifacts: import the rebuilt PPTX into Keynote, export PDF, then render at 144 dpi with
`pdftocairo -png -r 144`. The deck status date is supplied with `--status-date` when needed.
Presenter notes are emitted as JSON rather than embedded PowerPoint notes because `python-pptx`
notes XML failed Keynote import during compatibility testing.

Compile the LaTeX translation from `intraframe/results`:

```bash
mkdir -p /tmp/cognitive-gaze-latex
tectonic cognitive_gaze_tracking_evidence_deck.tex --outdir /tmp/cognitive-gaze-latex
cp /tmp/cognitive-gaze-latex/cognitive_gaze_tracking_evidence_deck.pdf \
  cognitive_gaze_tracking_evidence_deck_latex.pdf
```

The Beamer source references the reproducible PNGs in `cognitive_gaze_figures/` and the existing
result figures in the same directory. Compile it from `intraframe/results` so those relative paths
resolve.

Compile the expanded deck and paper from the same directory:

```bash
tectonic cognitive_gaze_tracking_explained_deck.tex --outdir .
tectonic cognitive_gaze_tracking_evidence_paper.tex --outdir .
```

## Verification performed

- JSON parsing and arithmetic consistency checks.
- Ruff and Python bytecode compilation for both generators.
- PowerPoint ZIP integrity and `python-pptx` reload.
- Keynote import and PDF export.
- PDF-to-PNG rendering of every slide.
- Contact-sheet and focused visual inspection.
- Fresh Tectonic builds and independent evidence-consistency review of the expanded deck and paper.
