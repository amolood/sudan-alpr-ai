#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — label the plate crops (semi-automatic).

For every crop in dataset/plates/ it shows the current model's best guess and
lets you accept it (Enter) or type the correct plate text. This turns labelling
from "type everything" into "fix the few that are wrong", which is much faster.

The label is the Latin serial of the plate, e.g.  7KH10346  (state code + digits,
no spaces, uppercase). Press 's' to skip a bad/blurry crop.

Output: dataset/labels.csv  with columns image_path,plate_text
        (the format fast-plate-ocr's trainer expects).

Usage:
    python 2_make_labels.py
"""

from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Label the merged, training-usable crops by default (override with $PLATES_DIR).
PLATES_DIR = os.environ.get(
    "PLATES_DIR",
    os.path.abspath(os.path.join(HERE, "..", "dataset", "good_plates")))
CSV_PATH = os.path.abspath(os.path.join(HERE, "..", "dataset", "labels.csv"))

# reuse the tuned two-column reader from the main recognizer for the pre-fill
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))


def guess(plate_bgr, alpr, cv2) -> str:
    from recognize import read_sudan_plate
    text, _ = read_sudan_plate(plate_bgr, alpr.ocr, cv2)
    return text


def main() -> int:
    if not os.path.isdir(PLATES_DIR):
        print("No crops found — run 1_crop_plates.py first.", file=sys.stderr)
        return 1
    crops = [f for f in sorted(os.listdir(PLATES_DIR)) if f.lower().endswith(".png")]
    if not crops:
        print("No crops to label.", file=sys.stderr)
        return 1

    import cv2
    from fast_alpr import ALPR
    alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-xs-v2-global-model")

    # keep any labels already done so re-runs are resumable
    done = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[os.path.basename(row["image_path"])] = row["plate_text"]

    import subprocess
    todo = [n for n in crops if n not in done]
    print(f"\n{len(done)} already labelled, {len(todo)} to go.")
    print("For each plate: a preview opens. Press Enter to accept the guess,")
    print("type the correct serial (e.g. 7KH10346), or 's' to skip.\n")

    rows = [(os.path.join(PLATES_DIR, n), done[n]) for n in crops if n in done]
    viewer = None
    for idx, name in enumerate(todo, 1):
        path = os.path.join(PLATES_DIR, name)
        g = guess(cv2.imread(path), alpr, cv2)
        # open an enlarged preview so the digits are readable while typing
        try:
            if viewer:
                viewer.terminate()
            viewer = subprocess.Popen(["qlmanage", "-p", path],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        except Exception:
            pass
        print(f"[{idx}/{len(todo)}] {name}")
        print(f"   model guess: {g or '(none)'}")
        ans = input("   correct text [Enter=accept, 's'=skip]: ").strip().upper()
        if ans == "S":
            continue
        label = ans if ans else g
        if label:
            rows.append((path, label))
        # save after every label so progress is never lost
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "plate_text"])
            for p, lb in rows:
                w.writerow([p, lb])
    if viewer:
        viewer.terminate()

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        for path, label in rows:
            w.writerow([path, label])

    print(f"\n✓ {len(rows)} labels written to {CSV_PATH}")
    print("Next: python 3_build_dataset.py  then  bash 4_train.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
