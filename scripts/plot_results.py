#!/usr/bin/env python3
"""Render the results comparison figure from committed manifests in results/.

Usage:
    python scripts/plot_results.py [--out figures/test1_summary.png]

Reads results/test1_flyback10ms.json and results/test1_active40ms.json only;
no video data or SDSLO_DATA_ROOT required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TIMINGS = ["flyback10ms", "active40ms"]
COLORS = {"flyback10ms": "#2b6cb0", "active40ms": "#c05621"}

# Sample roles ordered for the stacked bar; flyback samples are predicted
# (dead time), not observed motion.
ROLES = [
    ("observed_active", "observed (active)", "#2f855a"),
    ("observed_reacq", "observed (reacq)", "#68d391"),
    ("predicted_flyback", "predicted (flyback gap)", "#a0aec0"),
    ("rejected_mislock", "rejected (mislock)", "#e53e3e"),
    ("unlocked", "unlocked", "#1a202c"),
]


def load(timing: str) -> dict:
    with open(RESULTS / f"test1_{timing}.json") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIGURES / "test1_summary.png"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    data = {t: load(t) for t in TIMINGS}
    strip_hz = data["flyback10ms"]["strip_hz"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.suptitle(
        f"test1 pursuit raster — strip rate {strip_hz:.1f} Hz, "
        "two hardware timing assumptions",
        fontsize=11,
    )

    # Panel A: precision floor (RMS, arcmin)
    ax = axes[0]
    width = 0.35
    x = np.arange(2)
    for i, t in enumerate(TIMINGS):
        p = data[t]["precision_arcmin"]
        vals = [p["rms_x"], p["rms_y"]]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=t, color=COLORS[t])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, ["horizontal", "vertical"])
    ax.set_ylabel("RMS precision floor (arcmin)")
    ax.set_title("Precision floor (observed samples)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 4.6)

    # Panel B: pursuit-dot correlation
    ax = axes[1]
    for i, t in enumerate(TIMINGS):
        d = data[t]["dot_correlation"]
        vals = [d["r_x"], d["r_y"]]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=t, color=COLORS[t])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, ["r_x", "r_y"])
    ax.set_ylabel("Pearson r vs pursuit dot")
    ax.set_title("Pursuit-dot correlation", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="0.8", lw=0.8, ls="--")
    ax.legend(fontsize=8, loc="lower right")

    # Panel C: sample-role breakdown (fractions of all strip samples)
    ax = axes[2]
    y = np.arange(len(TIMINGS))
    for j, t in enumerate(TIMINGS):
        counts = data[t]["role_counts"]
        total = sum(counts.values())
        left = 0.0
        for key, label, color in ROLES:
            frac = counts.get(key, 0) / total
            ax.barh(j, frac, left=left, color=color,
                    label=label if j == 0 else None)
            if frac > 0.08:
                ax.text(left + frac / 2, j, f"{frac:.0%}", ha="center",
                        va="center", fontsize=7.5, color="white")
            left += frac
    ax.set_yticks(y, TIMINGS)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of strip samples")
    ax.set_title("Sample roles (flyback = predicted, not observed)", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.invert_yaxis()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
