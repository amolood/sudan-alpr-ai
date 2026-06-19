#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan ALPR — read plates from a video file or a live camera.

Runs the same pipeline as recognize_trained.py (fine-tuned OCR + the sudan_plate
interpreter) on every Nth frame of a video stream, and keeps a running tally of
which plates it has seen so a plate that passes by is reported once, not once
per frame.

Usage:
    # a video file
    ./venv/bin/python recognize_video.py path/to/clip.mp4

    # the default webcam
    ./venv/bin/python recognize_video.py 0

    # tune it
    ./venv/bin/python recognize_video.py clip.mp4 --every 5 --save out.mp4

Press 'q' in the preview window to stop.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OCR_MODEL = os.path.join(HERE, "models", "sudan_ocr.onnx")
PLATE_CFG = os.path.join(HERE, "models", "sudan_plate.yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sudan ALPR — video / camera reader")
    ap.add_argument("source", help="video file path, or a camera index like 0")
    ap.add_argument("--every", type=int, default=5,
                    help="run detection on every Nth frame (default 5; higher = faster)")
    ap.add_argument("--save", metavar="FILE",
                    help="also write the annotated video to FILE (e.g. out.mp4)")
    ap.add_argument("--no-window", action="store_true",
                    help="don't open a preview window (headless)")
    args = ap.parse_args()

    import cv2
    from fast_alpr import ALPR
    from sudan_plate import interpret

    # A camera index ("0") vs a file path.
    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not os.path.exists(source):
        print(f"file not found: {source}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"could not open video source: {source}", file=sys.stderr)
        return 1

    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model_path=OCR_MODEL,
        ocr_config_path=PLATE_CFG,
    )

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    seen: Counter[str] = Counter()   # plate text -> times confirmed
    last_label: dict[str, str] = {}  # plate text -> "Sudan / Khartoum [private]"
    frame_no = 0

    print("Reading… press 'q' in the window to stop (or Ctrl-C).")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1

            # Only run the heavy detection on every Nth frame; reuse the last
            # annotated overlay on the frames in between for a smooth preview.
            if frame_no % args.every == 0:
                drawn = alpr.draw_predictions(frame)
                frame = drawn.image
                for r in drawn.results:
                    info = interpret(r.ocr.text if r.ocr else "")
                    if not info.text:
                        continue
                    seen[info.text] += 1
                    if info.plate_class == "private":
                        loc = f"{info.state}"
                    else:
                        loc = info.plate_class_en
                    label = f"{info.country} / {loc}" if info.is_sudan else info.country
                    # Print a plate the first time we're confident about it.
                    if seen[info.text] == 2:
                        print(f"  🔖 {info.text:<12} {label}")
                    last_label[info.text] = label

            if writer is not None:
                writer.write(frame)
            if not args.no_window:
                cv2.imshow("Sudan ALPR — press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    # Summary: the plates we saw, most frequent first.
    print("\nPlates seen (by frame count):")
    if not seen:
        print("  — none")
    for text, count in seen.most_common():
        print(f"  {text:<12} ×{count:<4} {last_label.get(text, '')}")
    if args.save:
        print(f"\n✓ annotated video written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
