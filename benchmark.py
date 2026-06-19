#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan ALPR — OCR benchmark (educational).

Measures how accurately each OCR model reads Sudanese plates, by running it on
the *pre-cropped* plate images in `training/dataset/good_plates/` and comparing
the prediction against the human-verified ground truth in
`training/dataset/labels.csv`.

We benchmark the OCR stage in isolation (on already-cropped plates) so the
numbers reflect *reading* accuracy, not detector accuracy. That makes the two
models directly comparable on identical inputs.

Two models are compared:
    • global    : the off-the-shelf `cct-xs-v2-global-model` (65+ countries).
                  This is the baseline — it was never trained on Sudan.
    • sudan      : our fine-tuned model (`models/sudan_ocr.onnx`), trained on
                  real Sudanese plates.

Metrics reported (per model):
    • Exact-match accuracy : prediction == ground truth, character-for-character.
    • Character error rate : Levenshtein edit distance / total ground-truth chars
                             (CER; lower is better — captures "almost right").
    • Throughput           : plates read per second + mean ms/plate.

Usage
-----
    ./venv/bin/python benchmark.py                 # all models, full label set
    ./venv/bin/python benchmark.py --split val     # only the held-out val set
    ./venv/bin/python benchmark.py --models sudan  # one model only
    ./venv/bin/python benchmark.py --show-errors   # print every wrong read
    ./venv/bin/python benchmark.py --json out.json # also save machine-readable

Why this matters (education)
----------------------------
A benchmark is the honest scoreboard of an ML system. Without it, "it works"
is an opinion. With it, you can prove the fine-tuned model beats the baseline,
spot which plates fail, and track whether a change helped or hurt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "training", "dataset")
LABELS_CSV = os.path.join(DATASET, "labels.csv")
VAL_CSV = os.path.join(DATASET, "val.csv")
PLATES_DIR = os.path.join(DATASET, "good_plates")

SUDAN_OCR = os.path.join(HERE, "models", "sudan_ocr.onnx")
SUDAN_CFG = os.path.join(HERE, "models", "sudan_plate.yaml")
GLOBAL_HUB = "cct-xs-v2-global-model"


# ----------------------------------------------------------------------------
# Text utilities
# ----------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Canonical form for fair comparison: uppercase alphanumerics only.

    Ground truth uses no spaces (e.g. "7KH10346"); model output may include
    spaces, pad chars, or lowercase. We strip all of that so the comparison
    measures *reading* accuracy, not formatting.
    """
    return "".join(c for c in text.upper() if c.isalnum())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert/delete/substitute = cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


# ----------------------------------------------------------------------------
# Ground-truth loading
# ----------------------------------------------------------------------------

def load_labels(split: str) -> list[tuple[str, str]]:
    """Return [(absolute_image_path, ground_truth_text), ...] for the split.

    split="all" -> labels.csv (every verified plate)
    split="val" -> val.csv    (the held-out validation plates only)
    """
    csv_path = VAL_CSV if split == "val" else LABELS_CSV
    if not os.path.exists(csv_path):
        sys.exit(f"missing ground-truth file: {csv_path}")

    items: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_path = row["image_path"].strip()
            gt = normalize(row["plate_text"])
            if not gt:
                continue
            # Paths in labels.csv are absolute; in val.csv they're relative to
            # the dataset dir. Resolve both shapes and skip missing files.
            path = raw_path if os.path.isabs(raw_path) else os.path.join(DATASET, raw_path)
            if not os.path.exists(path):
                base = os.path.basename(raw_path)
                alt = os.path.join(PLATES_DIR, base)
                path = alt if os.path.exists(alt) else path
            if os.path.exists(path):
                items.append((path, gt))
    return items


# ----------------------------------------------------------------------------
# Model wrappers
# ----------------------------------------------------------------------------

def build_recognizer(name: str):
    """Return a fast_plate_ocr.LicensePlateRecognizer for the given model name."""
    from fast_plate_ocr import LicensePlateRecognizer

    if name == "global":
        return LicensePlateRecognizer(hub_ocr_model=GLOBAL_HUB)
    if name == "sudan":
        if not os.path.exists(SUDAN_OCR):
            sys.exit(f"fine-tuned model not found: {SUDAN_OCR}")
        return LicensePlateRecognizer(
            onnx_model_path=SUDAN_OCR,
            plate_config_path=SUDAN_CFG,
        )
    sys.exit(f"unknown model: {name} (choose from: global, sudan)")


# ----------------------------------------------------------------------------
# Benchmark core
# ----------------------------------------------------------------------------

def evaluate(name: str, items: list[tuple[str, str]], show_errors: bool):
    """Run one model over all items and return a metrics dict."""
    rec = build_recognizer(name)

    correct = 0
    total_edits = 0
    total_gt_chars = 0
    errors: list[dict] = []

    # Warm up once (first inference includes graph init / allocation) so the
    # timing below reflects steady-state speed, not one-off startup cost.
    rec.run_one(items[0][0])

    t0 = time.perf_counter()
    for path, gt in items:
        pred = normalize(rec.run_one(path).plate)
        dist = edit_distance(pred, gt)
        total_edits += dist
        total_gt_chars += len(gt)
        if pred == gt:
            correct += 1
        elif show_errors:
            errors.append({"image": os.path.basename(path),
                           "expected": gt, "got": pred, "edits": dist})
    elapsed = time.perf_counter() - t0

    n = len(items)
    return {
        "model": name,
        "plates": n,
        "exact_match": correct,
        "accuracy": correct / n if n else 0.0,
        "cer": total_edits / total_gt_chars if total_gt_chars else 0.0,
        "seconds": elapsed,
        "ms_per_plate": (elapsed / n * 1000) if n else 0.0,
        "plates_per_sec": (n / elapsed) if elapsed else 0.0,
        "errors": errors,
    }


def evaluate_country(name: str, items: list[tuple[str, str]], show_errors: bool):
    """Measure country/state recognition end-to-end.

    For each plate we run the OCR model, feed its text to the Sudan interpreter,
    and check that it (a) flags the plate as Sudanese and (b) names the right
    state. Ground truth here: every plate in our dataset *is* Sudanese, and its
    state is the letters in labels.csv — so this measures the recall of the
    country+state recogniser on real (imperfect) OCR output.
    """
    from sudan_plate import interpret, CODE_ALIASES

    rec = build_recognizer(name)

    def gt_state(gt: str) -> str:
        """Pull the state code out of a clean ground-truth string like 7KH10346."""
        letters = "".join(c for c in gt if c.isalpha())
        return CODE_ALIASES.get(letters, letters)

    sudan_ok = state_ok = total = 0
    errors: list[dict] = []
    rec.run_one(items[0][0])  # warm up
    for path, gt in items:
        pred_text = rec.run_one(path).plate
        info = interpret(pred_text)
        total += 1
        if info.is_sudan:
            sudan_ok += 1
        if info.is_sudan and info.state_code == gt_state(gt):
            state_ok += 1
        elif show_errors:
            errors.append({"image": os.path.basename(path), "gt": gt,
                           "got_country": info.country, "got_state": info.state_code})
    return {
        "model": name,
        "plates": total,
        "sudan_recognized": sudan_ok,
        "sudan_accuracy": sudan_ok / total if total else 0.0,
        "state_correct": state_ok,
        "state_accuracy": state_ok / total if total else 0.0,
        "errors": errors,
    }


def print_country_report(results: list[dict], split: str):
    n = results[0]["plates"] if results else 0
    print(f"\nSudan ALPR — country + state recognition   (split={split}, {n} plates)\n")
    header = f"{'model':<8} {'is-Sudan':>14} {'sudan-acc':>10} {'state-correct':>14} {'state-acc':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<8} "
              f"{r['sudan_recognized']:>5}/{r['plates']:<5}    "
              f"{r['sudan_accuracy']*100:>7.1f}%   "
              f"{r['state_correct']:>5}/{r['plates']:<5}    "
              f"{r['state_accuracy']*100:>7.1f}%")
    for r in results:
        if r["errors"]:
            print(f"\n--- {r['model']} country/state misses ({len(r['errors'])}) ---")
            for e in r["errors"]:
                print(f"  {e['image']:<28} gt {e['gt']:<10} "
                      f"-> {e['got_country']}/{e['got_state'] or '-'}")


def print_report(results: list[dict], split: str):
    n = results[0]["plates"] if results else 0
    print(f"\nSudan ALPR — OCR benchmark   (split={split}, {n} plates)\n")
    header = f"{'model':<8} {'exact':>10} {'accuracy':>9} {'CER':>7} {'ms/plate':>9} {'plates/s':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<8} "
              f"{r['exact_match']:>4}/{r['plates']:<5} "
              f"{r['accuracy']*100:>7.1f}% "
              f"{r['cer']*100:>6.1f}% "
              f"{r['ms_per_plate']:>8.1f} "
              f"{r['plates_per_sec']:>9.1f}")

    # Highlight the improvement if both models ran.
    by_name = {r["model"]: r for r in results}
    if "global" in by_name and "sudan" in by_name:
        g, s = by_name["global"], by_name["sudan"]
        delta = (s["accuracy"] - g["accuracy"]) * 100
        print(f"\nFine-tuning lifted exact-match accuracy by {delta:+.1f} "
              f"percentage points over the global baseline.")

    for r in results:
        if r["errors"]:
            print(f"\n--- {r['model']} misreads ({len(r['errors'])}) ---")
            for e in r["errors"]:
                print(f"  {e['image']:<28} expected {e['expected']:<10} "
                      f"got {e['got']:<10} (edits={e['edits']})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sudan ALPR OCR benchmark")
    ap.add_argument("--split", choices=["all", "val"], default="all",
                    help="'all' = every labelled plate; 'val' = held-out set only")
    ap.add_argument("--models", nargs="+", default=["global", "sudan"],
                    choices=["global", "sudan"],
                    help="which OCR models to benchmark")
    ap.add_argument("--country", action="store_true",
                    help="benchmark country (is-Sudan) + state recognition "
                         "instead of raw OCR text accuracy")
    ap.add_argument("--show-errors", action="store_true",
                    help="list every misread plate")
    ap.add_argument("--json", metavar="FILE",
                    help="also write the metrics as JSON to FILE")
    args = ap.parse_args()

    items = load_labels(args.split)
    if not items:
        sys.exit("no labelled plates found — did the dataset get built?")

    print(f"Loaded {len(items)} ground-truth plates from "
          f"{'val.csv' if args.split == 'val' else 'labels.csv'}.")

    results = []
    for name in args.models:
        print(f"Running '{name}' …", flush=True)
        if args.country:
            results.append(evaluate_country(name, items, args.show_errors))
        else:
            results.append(evaluate(name, items, args.show_errors))

    if args.country:
        print_country_report(results, args.split)
    else:
        print_report(results, args.split)

    if args.json:
        # Drop the verbose error lists from the saved summary unless requested.
        payload = {"split": args.split, "plates": len(items), "models": results}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
