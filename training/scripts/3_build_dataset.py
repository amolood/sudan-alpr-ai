#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 — merge real + synthetic labels and split into train / val CSVs.

Combines:
    dataset/labels.csv          (real, hand-corrected plates — high value)
    dataset/labels_synth.csv    (synthetic plates — volume)
and writes train.csv / val.csv that the trainer consumes.

Real plates are duplicated a few times (--real-weight) so the model doesn't
drown them in synthetic data, and ALL real plates are kept in BOTH the training
signal and the validation set is drawn so it contains real plates (the thing we
actually care about reading).

Usage:
    python 4_build_dataset.py --real-weight 8 --val-frac 0.1
"""

from __future__ import annotations

import argparse
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, "..", "dataset"))


def read_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r["image_path"]
            # The trainer resolves image_path relative to the CSV's folder (DS),
            # so store paths relative to DS, not absolute (avoids dataset//abs).
            if os.path.isabs(p):
                p = os.path.relpath(p, DS)
            rows.append((p, r["plate_text"]))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-weight", type=int, default=8,
                    help="how many times to repeat each real plate in training")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    real = read_csv(os.path.join(DS, "labels.csv"))
    print(f"real plates: {len(real)}")
    if not real:
        print("No labelled plates. Run 1_crop_plates.py then 2_make_labels.py first.")
        return 1
    if len(real) < 200:
        print(f"\n⚠️  Only {len(real)} real plates. OCR training needs ~1000+ to")
        print("    generalise; below that the model memorises and won't read new")
        print("    plates. Keep collecting images into input/ and re-label.\n")

    random.shuffle(real)
    n_val = max(1, int(len(real) * args.val_frac))
    val = real[:n_val]
    train_real = real[n_val:]

    # repeat real plates so a small set still gives the optimiser enough steps
    train = train_real * args.real_weight
    random.shuffle(train)

    write_csv(os.path.join(DS, "train.csv"), train)
    write_csv(os.path.join(DS, "val.csv"), val)
    print(f"\n✓ train.csv: {len(train)} rows  (real x{args.real_weight})")
    print(f"✓ val.csv:   {len(val)} rows")
    print("Next: bash 4_train.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
