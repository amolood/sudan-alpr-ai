#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0 — auto-collect REAL Sudanese plate images from public web search.

Pipeline (fully automatic):
    1. SEARCH  : query public image engines (Bing/Google) for Sudan plate photos.
    2. DETECT  : keep only images where the YOLO detector finds a plate.
    3. VERIFY  : keep only plates whose header reads "SUDAN" (drops the Saudi /
                 Jordanian / other look-alikes that general search mixes in).

It uses GENERAL search engines only — it does NOT scrape Facebook or any
login-gated / ToS-protected site.

Outputs:
    dataset/web_raw/        all downloaded images (kept for reference)
    dataset/web_plates/     verified Sudanese plate crops, ready to label

Usage:
    python 0_fetch_web_images.py --per-query 80
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "dataset", "web_raw"))
OUT = os.path.abspath(os.path.join(HERE, "..", "dataset", "web_plates"))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

QUERIES = [
    # round 1 (already used) — kept for resumability
    "لوحة سيارة سودانية",
    "لوحات السيارات السودانية الجديدة",
    "Sudan car license plate",
    "Sudan vehicle number plate Khartoum",
    # round 2 — fresh queries by state / brand / context to surface NEW photos
    "لوحة سيارة بحري السودان",
    "لوحة سيارة ام درمان",
    "لوحة سيارة بورتسودان",
    "لوحة سيارة مدني السودان",
    "لوحة سيارة كسلا السودان",
    "لوحة سيارة نيالا دارفور",
    "لوحة سيارة الجزيرة السودان",
    "لوحة سيارة كوستي السودان",
    "تويوتا سودان لوحة",
    "هيلوكس سودان لوحة",
    "لاندكروزر سودان لوحة رقم",
    "بوكس سودان لوحة سيارة",
    "Sudan Khartoum car number plate photo",
    "Sudan Toyota Hilux plate",
    "Sudan vehicle plate registration 2023",
    "Sudanese vehicle plate KH NS GZ",
    "عربية سودانية رقم لوحة",
    "بيع سيارات السودان لوحة",
    "سوق السيارات السودان لوحة",
    "Sudan used cars license plate",
]


def download(per_query: int):
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    os.makedirs(RAW, exist_ok=True)
    for i, q in enumerate(QUERIES):
        out = os.path.join(RAW, f"q{i:02d}")
        os.makedirs(out, exist_ok=True)
        print(f"🔎 {q}")
        for engine in (BingImageCrawler, GoogleImageCrawler):
            try:
                engine(downloader_threads=4, storage={"root_dir": out},
                       log_level=40).crawl(keyword=q, max_num=per_query)
            except Exception as e:
                print(f"   ({engine.__name__} failed: {e})")
            if os.listdir(out):
                break
        print(f"   -> {len(os.listdir(out))} files")


def filter_sudanese():
    import cv2
    from fast_alpr import ALPR
    alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-xs-v2-global-model")

    os.makedirs(OUT, exist_ok=True)
    raw = [f for f in glob.glob(os.path.join(RAW, "**", "*.*"), recursive=True)
           if f.lower().endswith(IMG_EXTS)]
    print(f"\nFiltering {len(raw)} downloaded images…")

    kept = 0
    for f in raw:
        img = cv2.imread(f)
        if img is None:
            continue
        dets = [d for d in alpr.detector.predict(img) if float(d.confidence) > 0.5]
        for i, d in enumerate(dets):
            bb = d.bounding_box
            x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
            x2, y2 = int(bb.x2), int(bb.y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # verify the "SUDAN" header so look-alike foreign plates are dropped
            H, W = crop.shape[:2]
            top = crop[: int(H * 0.45), :]
            top = cv2.resize(top, (top.shape[1] * 3, top.shape[0] * 3),
                             interpolation=cv2.INTER_CUBIC)
            header = (alpr.ocr.predict(top).text or "").upper()
            if not any(k in header for k in ("SUDAN", "SDAN", "SUDN", "UDAN")):
                continue
            kept += 1
            cv2.imwrite(os.path.join(OUT, f"web_{kept:04d}.png"), crop)

    print(f"\n✓ {kept} verified Sudanese plate crops -> {OUT}")
    print("Next: label them with 2_make_labels.py (point it at web_plates/).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=80)
    ap.add_argument("--skip-download", action="store_true",
                    help="only re-run the filter over already-downloaded images")
    args = ap.parse_args()

    if not args.skip_download:
        download(args.per_query)
    filter_sudanese()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
