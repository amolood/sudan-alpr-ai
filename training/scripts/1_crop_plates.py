#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — crop license plates out of full car photos.

Runs the YOLO plate detector (the same one FastALPR ships) over every image in
a source folder and saves each detected plate as its own image under
dataset/plates/. These crops are what you then label in step 2.

Usage:
    python 1_crop_plates.py ../../input            # crop from the input folder
    python 1_crop_plates.py /path/to/more/photos   # add more later

Crops are named <source>__<index>.png so you can trace each back to its photo.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLATES_DIR = os.path.abspath(os.path.join(HERE, "..", "dataset", "plates"))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "input")
    src = os.path.abspath(src)

    images = []
    if os.path.isdir(src):
        for n in sorted(os.listdir(src)):
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                images.append(os.path.join(src, n))
    elif os.path.isfile(src):
        images = [src]
    if not images:
        print(f"No images found at {src}", file=sys.stderr)
        return 1

    import cv2
    from fast_alpr import ALPR

    os.makedirs(PLATES_DIR, exist_ok=True)
    alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-xs-v2-global-model")

    total = 0
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0].replace(" ", "_")
        for i, det in enumerate(alpr.detector.predict(frame)):
            bb = det.bounding_box
            x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
            x2, y2 = int(bb.x2), int(bb.y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out = os.path.join(PLATES_DIR, f"{stem}__{i}.png")
            cv2.imwrite(out, crop)
            total += 1
            print(f"  saved {os.path.basename(out)}  ({x2-x1}x{y2-y1}px)")

    print(f"\n✓ {total} plate crops written to {PLATES_DIR}")
    print("Next: run  2_make_labels.py  to label them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
