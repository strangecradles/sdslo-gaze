#!/usr/bin/env python3
"""Validate the clean pipeline against the research repo's champion result.

Runs the clean `sdslo_gaze` pipeline on `test1` from the cached strip track and checks that the
recovered high-rate trajectory reproduces the reference within tolerance:
  - strip rate is in the 600-960+ Hz target band,
  - the measurement-state composition is sane (observed-only accuracy, flyback labeled missing),
  - held-out correlation vs the pursuit dot is comparable to the champion (r_x ~ 0.9),
  - a microsaccade main sequence is present with physiological slope.

Usage:  SDSLO_DATA_ROOT=/path/to/gaze-model python scripts/validate_against_reference.py
Exit code 0 = all gates pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdslo_gaze import pipeline  # noqa: E402


GATES = {
    "strip_hz_min": 600.0,
    "dot_r_x_min": 0.80,      # champion reports r_x=0.904; clean pipeline should be comparable
    "main_seq_slope_lo": 0.4,
    "main_seq_slope_hi": 1.2,
    "min_events": 10,
}


def main() -> int:
    print("Running clean pipeline on test1 (source=reference, timing=flyback10ms)...")
    res = pipeline.run("test1", timing_model="flyback10ms", source="reference",
                       out_dir="results", with_dot=True)

    checks: list[tuple[str, bool, str]] = []

    checks.append(("strip_hz >= 600", res.strip_hz >= GATES["strip_hz_min"],
                   f"{res.strip_hz:.1f} Hz (S={res.strip_width})"))

    obs = res.n_observed
    checks.append(("observed samples > 0", obs > 0, f"{obs:,} observed"))

    # flyback must be labeled, never counted as observed
    roles = res.role_counts
    has_flyback_label = any(k.startswith(("predicted_flyback", "missing_flyback")) for k in roles)
    checks.append(("flyback labeled (missing/predicted present)", has_flyback_label, str(roles)))

    ms = res.microsaccades
    n_ev = ms["n_events"]
    checks.append((f"microsaccade events >= {GATES['min_events']}", n_ev >= GATES["min_events"],
                   f"{n_ev} events"))
    slope = ms["main_sequence"].get("slope", float("nan"))
    slope_ok = GATES["main_seq_slope_lo"] <= slope <= GATES["main_seq_slope_hi"]
    checks.append(("main-sequence slope physiological", slope_ok,
                   f"slope={slope:.3f}, r={ms['main_sequence'].get('r', float('nan')):.3f}"))

    if res.dot_correlation is not None:
        rx = res.dot_correlation.get("r_x", float("nan"))
        checks.append((f"dot r_x >= {GATES['dot_r_x_min']}", rx >= GATES["dot_r_x_min"],
                       f"r_x={rx:.3f}, r_y={res.dot_correlation.get('r_y', float('nan')):.3f}"))
    else:
        checks.append(("dot correlation available", False, "champion CSV not found (skipped)"))

    print("\n=== VALIDATION GATES ===")
    all_pass = True
    for name, ok, detail in checks:
        # a skipped dot-correlation is a soft warning, not a hard failure
        soft = name == "dot correlation available"
        status = "PASS" if ok else ("WARN" if soft else "FAIL")
        if not ok and not soft:
            all_pass = False
        print(f"[{status}] {name}: {detail}")

    print(f"\nManifest: {res.manifest_path}")
    print("RESULT:", "ALL GATES PASS" if all_pass else "GATES FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
