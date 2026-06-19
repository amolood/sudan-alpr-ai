#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan ALPR (AI)  -  professional license-plate recognition with FastALPR.

Two deep-learning stages (this is how real ANPR works):
    1) DETECT  : a YOLO-v9 model finds the plate(s) in the photo.
    2) READ    : a CCT transformer OCR model reads the plate text.

Both models run locally via ONNX Runtime (CoreML-accelerated on Apple Silicon).
The OCR model is a global model trained on 65+ countries, so it reads the
Latin serial line of Sudanese plates (e.g. "3KH3476") robustly — even on
angled, real-world car photos where the old MATLAB template-matcher failed.

Usage
-----
    python recognize.py path/to/car.jpg            # one image
    python recognize.py input/                     # every image in a folder
    python recognize.py car.jpg --min-conf 0.5     # filter weak reads

Annotated images (plate boxed + text drawn) are written to ./output/.
Results are also printed and saved to output/results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather_images(path: str) -> list[str]:
    if os.path.isdir(path):
        out = []
        for name in sorted(os.listdir(path)):
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                out.append(os.path.join(path, name))
        return out
    return [path]


def mean_conf(ocr) -> float:
    """OcrResult.confidence is a per-character list; collapse to one number."""
    if ocr is None or not getattr(ocr, "confidence", None):
        return 0.0
    vals = [c for c in ocr.confidence if c is not None]
    return sum(vals) / len(vals) if vals else 0.0


# Letters that the OCR commonly mistakes for digits, and their digit form.
# Used to clean up the all-digit serial column of a Sudanese plate.
DIGIT_FIX = {
    "O": "0", "U": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1", "T": "1", "J": "1",
    "S": "5", "B": "8", "G": "6", "A": "4", "Z": "2",
    "_": "", "-": "", " ": "",
}


def _to_digits(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(ch)
        elif ch in DIGIT_FIX:
            out.append(DIGIT_FIX[ch])
    return "".join(out)


def _ocr_text(ocr_engine, crop, cv2, scale: int = 4) -> str:
    """Upscale a small crop and OCR it; returns the raw text (may be empty)."""
    if crop is None or crop.size == 0:
        return ""
    big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                     interpolation=cv2.INTER_CUBIC)
    res = ocr_engine.predict(big)
    return (res.text or "") if res else ""


def dominant_plate_color(plate_bgr) -> str:
    """Roughly classify a plate's background colour.

    Sudanese plate *class* tracks colour (red=army/police, yellow=government,
    green=commercial, silver/white=private…). We don't need precision — just a
    coarse hint the interpreter can use to corroborate the text marker. Returns
    one of: red, yellow, green, blue, silver, white, or "" if unsure.

    We look at the plate's border region (the centre is mostly characters), in
    HSV, and pick the colour whose hue/saturation band dominates.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return ""
    if plate_bgr is None or plate_bgr.size == 0:
        return ""

    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat_mean = float(s.mean())

    # Low saturation -> greyscale plate: silver vs white by brightness.
    if sat_mean < 45:
        return "white" if float(v.mean()) > 170 else "silver"

    # Otherwise classify by hue among the saturated pixels.
    mask = s > 60
    if mask.sum() < 0.05 * s.size:
        return "silver"
    hue = h[mask]
    # OpenCV hue is 0–179. Count pixels per colour band.
    bands = {
        "red":    int(((hue < 10) | (hue > 160)).sum()),
        "yellow": int(((hue >= 18) & (hue < 38)).sum()),
        "green":  int(((hue >= 38) & (hue < 85)).sum()),
        "blue":   int(((hue >= 90) & (hue < 130)).sum()),
    }
    best = max(bands, key=bands.get)
    return best if bands[best] > 0.15 * hue.size else "silver"


def read_sudan_plate(plate_bgr, ocr_engine, cv2):
    """
    Read a cropped Sudanese plate using its KNOWN two-column layout instead of
    OCR-ing the whole plate at once (which mixes the big Arabic line with the
    small Latin line and fails).

    Layout:
        ┌──────────────────────────┐
        │     SUDAN     السودان     │   top ~ third  (ignored)
        │  ٧ خ          ١٠٣٤٦       │   middle: big Arabic line
        │  7KH          10346       │   bottom: small Latin line  <-- we read this
        └──────────────────────────┘
        left column  = state code (7KH)   right column = serial digits (10346)

    Returns (text, info_dict).
    """
    H, W = plate_bgr.shape[:2]
    # Left column, bottom portion -> "7KH" (letters + leading state digit).
    state_crop = plate_bgr[int(H * 0.55):H, 0:int(W * 0.40)]
    state = _ocr_text(ocr_engine, state_crop, cv2)
    state = "".join(c for c in state if c.isalnum()).upper()

    # Right column, bottom Latin strip -> the serial digits.
    num_crop = plate_bgr[int(H * 0.70):int(H * 0.97), int(W * 0.40):W]
    num_raw = _ocr_text(ocr_engine, num_crop, cv2)
    num = _to_digits(num_raw)

    # Fallback: on tiny/blurry plates the column split can come back empty or
    # nonsensical (no serial digits, or a state code that isn't D+LL). In that
    # case read the whole plate at once instead — less precise, but better than
    # an empty result.
    looks_valid = bool(num) and len(state) >= 2
    if not looks_valid:
        whole = _ocr_text(ocr_engine, plate_bgr, cv2)
        whole = "".join(c for c in whole if c.isalnum()).upper()
        return whole, {"state": "", "serial": "", "serial_raw": whole,
                       "method": "whole-plate-fallback"}

    text = f"{state}{num}"
    return text, {"state": state, "serial": num, "serial_raw": num_raw,
                  "method": "column-split"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Sudan ALPR (AI) — FastALPR")
    ap.add_argument("source", help="image file or a folder of images")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output"),
                    help="folder for annotated images + results.json")
    ap.add_argument("--det-conf", type=float, default=0.3,
                    help="discard plate detections below this confidence")
    ap.add_argument("--detector", default="yolo-v9-t-384-license-plate-end2end")
    ap.add_argument("--ocr", default="cct-xs-v2-global-model")
    args = ap.parse_args()

    images = gather_images(args.source)
    if not images:
        print(f"No images found at: {args.source}", file=sys.stderr)
        return 1

    # Heavy imports are deferred so --help stays instant.
    import cv2
    from fast_alpr import ALPR

    from sudan_plate import interpret

    print("Loading models (first run downloads weights)…")
    alpr = ALPR(detector_model=args.detector, ocr_model=args.ocr)
    os.makedirs(args.out, exist_ok=True)

    all_results = []
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ! could not read {img_path}")
            continue

        # Stage 1: detect plates with the YOLO detector.
        detections = alpr.detector.predict(frame)

        annotated = frame.copy()
        plates = []
        for det in detections:
            if float(det.confidence) < args.det_conf:
                continue
            bb = det.bounding_box
            x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
            x2, y2 = int(bb.x2), int(bb.y2)
            plate_bgr = frame[y1:y2, x1:x2]
            if plate_bgr.size == 0:
                continue

            # Stage 2: read using the Sudanese two-column layout.
            text, info = read_sudan_plate(plate_bgr, alpr.ocr, cv2)

            # Stage 3: verify country, plate class, and state from the text,
            # using the plate's dominant colour as corroborating evidence.
            color = dominant_plate_color(plate_bgr)
            plate = interpret(text, color=color)

            plates.append({
                "text": plate.text,
                "country": plate.country,
                "country_confidence": plate.country_confidence,
                "is_sudan": plate.is_sudan,
                "plate_class": plate.plate_class,
                "plate_class_en": plate.plate_class_en,
                "color": plate.color,
                "state": plate.state,
                "state_code": plate.state_code,
                "serial": plate.serial or info["serial"],
                "detect_confidence": round(float(det.confidence), 3),
                "box": [x1, y1, x2, y2],
            })

            # draw box + recognised text (with class/state when Sudanese)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = text if text else "?"
            if plate.is_sudan and plate.plate_class == "private":
                label = f"{label}  {plate.state_code}/Sudan"
            elif plate.is_sudan:
                label = f"{label}  {plate.plate_class_en}/Sudan"
            cv2.putText(annotated, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

        out_img = os.path.join(args.out, "annotated_" + os.path.basename(img_path))
        cv2.imwrite(out_img, annotated)

        all_results.append({"image": img_path, "annotated": out_img, "plates": plates})

        print(f"\n📷 {os.path.basename(img_path)}")
        if plates:
            for p in plates:
                flag = "🇸🇩" if p["is_sudan"] else "🌐"
                if p["is_sudan"] and p["plate_class"] == "private":
                    loc = f"{p['country']} / {p['state']}"
                elif p["is_sudan"]:
                    loc = f"{p['country']} / {p['plate_class_en']}"
                else:
                    loc = p["country"]
                print(f"    🔖 {p['text']:<12} {flag} {loc:<26} "
                      f"(country {p['country_confidence']*100:.0f}% | "
                      f"serial={p['serial']}  detect {p['detect_confidence']*100:.0f}%)")
        else:
            print("    — no plate detected above the confidence threshold")

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Annotated images + results.json written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
