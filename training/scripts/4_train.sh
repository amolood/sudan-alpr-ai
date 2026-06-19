#!/usr/bin/env bash
# Step 4 — FINE-TUNE the Sudanese plate OCR model from the global pretrained
# weights (NOT from scratch). The global cct-xs-v2 model already knows letters
# and digits across 65+ countries; we only adapt it to the Sudanese plate's
# look. This is the key fix over the earlier from-scratch attempt that failed.
#
# Usage:
#   bash 4_train.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/venv/bin/fast-plate-ocr"
DS="$HERE/../dataset"
CFG="$HERE/../config"
OUT="$HERE/../output"
WEIGHTS="$CFG/cct_xs_v2_global.keras"   # pretrained global weights

mkdir -p "$OUT"

KERAS_BACKEND=tensorflow "$PY" train \
  --model-config-file "$CFG/cct_xs_v2_model.yaml" \
  --plate-config-file "$CFG/sudan_plate.yaml" \
  --annotations "$DS/train.csv" \
  --val-annotations "$DS/val.csv" \
  --weights-path "$WEIGHTS" \
  --validate-dataset warn \
  --epochs 120 \
  --batch-size 32 \
  --lr 0.0003 \
  --early-stopping-patience 25 \
  --early-stopping-metric val_plate_acc \
  --output-dir "$OUT"

echo ""
echo "✓ Fine-tuning done. Model saved under $OUT"
