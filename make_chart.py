#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the benchmark chart shown in the README (docs/benchmark.png).

Reads numbers straight from output/benchmark.json when it exists, so the chart
stays in sync with whatever `benchmark.py --json output/benchmark.json` last
produced. Falls back to the documented numbers if the file isn't there.

    ./venv/bin/python make_chart.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")  # no display needed; write straight to a file
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_JSON = os.path.join(HERE, "output", "benchmark.json")
OUT = os.path.join(HERE, "docs", "benchmark.png")

# Defaults match the README; overwritten by benchmark.json if present.
acc = {"global": 0.0, "sudan": 82.6}
cer = {"global": 78.3, "sudan": 7.9}

if os.path.exists(BENCH_JSON):
    data = json.load(open(BENCH_JSON))
    for m in data.get("models", []):
        acc[m["model"]] = round(m["accuracy"] * 100, 1)
        cer[m["model"]] = round(m["cer"] * 100, 1)

labels = ["global\n(off-the-shelf)", "sudan\n(fine-tuned)"]
acc_vals = [acc["global"], acc["sudan"]]
cer_vals = [cer["global"], cer["sudan"]]
colors = ["#9aa0a6", "#2e7d32"]  # grey baseline, green fine-tuned

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
fig.suptitle("Sudan ALPR — OCR benchmark (121 labeled plates)",
             fontsize=13, fontweight="bold")

# Left: exact-match accuracy (higher is better)
bars1 = ax1.bar(labels, acc_vals, color=colors, width=0.55)
ax1.set_title("Exact-match accuracy  (higher is better)", fontsize=10)
ax1.set_ylabel("%")
ax1.set_ylim(0, 100)
ax1.bar_label(bars1, fmt="%.1f%%", padding=3, fontweight="bold")
ax1.grid(axis="y", alpha=0.25)

# Right: character error rate (lower is better)
bars2 = ax2.bar(labels, cer_vals, color=colors, width=0.55)
ax2.set_title("Character error rate  (lower is better)", fontsize=10)
ax2.set_ylabel("%")
ax2.set_ylim(0, 100)
ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontweight="bold")
ax2.grid(axis="y", alpha=0.25)

for ax in (ax1, ax2):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
