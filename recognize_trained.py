#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan ALPR — recognition using the FINE-TUNED Sudanese OCR model.

Pipeline:
    1. DETECT : YOLO-v9 plate detector (FastALPR's, unchanged).
    2. READ   : our own OCR model, fine-tuned from the global cct-xs-v2 weights
                on real Sudanese plates -> reads "7KH10346" style serials.

This replaces the hand-tuned two-column reader in recognize.py. On held-out
real plates the fine-tuned model scored ~84% exact-match.

Usage:
    ./venv/bin/python recognize_trained.py input/
    ./venv/bin/python recognize_trained.py input/some_car.jpg
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OCR_MODEL = os.path.join(HERE, "models", "sudan_ocr.onnx")
PLATE_CFG = os.path.join(HERE, "models", "sudan_plate.yaml")
OUT_DIR = os.path.join(HERE, "output")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather(path):
    if os.path.isdir(path):
        return [os.path.join(path, n) for n in sorted(os.listdir(path))
                if os.path.splitext(n)[1].lower() in IMG_EXTS]
    return [path]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: recognize_trained.py <image|folder>", file=sys.stderr)
        return 1
    images = gather(sys.argv[1])
    if not images:
        print("no images found", file=sys.stderr)
        return 1

    import cv2
    from fast_alpr import ALPR

    from sudan_plate import interpret

    # Detector from the hub; OCR is OUR fine-tuned Sudanese model.
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model_path=OCR_MODEL,
        ocr_config_path=PLATE_CFG,
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        drawn = alpr.draw_predictions(frame)
        out_img = os.path.join(OUT_DIR, "trained_" + os.path.basename(img_path))
        cv2.imwrite(out_img, drawn.image)

        plates = []
        for r in drawn.results:
            text = r.ocr.text if r.ocr else ""
            info = interpret(text)
            plates.append({"text": info.text,
                           "country": info.country,
                           "country_confidence": info.country_confidence,
                           "state": info.state,
                           "state_code": info.state_code,
                           "is_sudan": info.is_sudan,
                           "detect_conf": round(float(r.detection.confidence), 3)})
        results.append({"image": img_path, "plates": plates})

        print(f"\n📷 {os.path.basename(img_path)}")
        for p in plates:
            flag = "🇸🇩" if p["is_sudan"] else "🌐"
            loc = f"{p['country']}" + (f" / {p['state']}" if p["is_sudan"] else "")
            print(f"    🔖 {p['text']:<12} {flag} {loc:<22} "
                  f"(country {p['country_confidence']*100:.0f}% | detect {p['detect_conf']*100:.0f}%)")
        if not plates:
            print("    — no plate detected")

    with open(os.path.join(OUT_DIR, "trained_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ results + annotated images in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
