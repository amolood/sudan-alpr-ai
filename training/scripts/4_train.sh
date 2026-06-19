#!/usr/bin/env bash
# Step 4 — FINE-TUNE the Sudanese plate OCR model with WandB metrics tracking.
#
# Usage:
#   WANDB_API_KEY="your_key_here" bash 4_train.sh

set -euo pipefail

# --- Environment Setup ---
HERE="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/venv/bin/fast-plate-ocr"
DS="$HERE/../dataset"
CFG="$HERE/../config"
OUT="$HERE/../output"
WEIGHTS="$CFG/cct_xs_v2_global.keras"

# --- Signal Trapping ---
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [ $exit_code -ne 0 ]; then
        echo "Error: Training interrupted or failed unexpectedly." >&2
    fi
    exit $exit_code
}
trap cleanup EXIT INT TERM

# --- Pre-flight Checks ---
echo "Running pre-flight structural checks..."
for tool in "$PY" "$WEIGHTS" "$DS/train.csv" "$DS/val.csv" "$CFG/cct_xs_v2_model.yaml" "$CFG/sudan_plate.yaml"; do
    if [ ! -e "$tool" ]; then
        echo "Critical Error: Target path missing -> $tool" >&2
        exit 1
    fi
done

# Ensure wandb Python package is available in the venv
if ! "$ROOT/venv/bin/python" -c "import wandb" &>/dev/null; then
    echo "WandB package missing. Installing into virtual environment..."
    "$ROOT/venv/bin/pip" install wandb
fi

mkdir -p "$OUT"

# --- Resource Auto-Detection ---
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_WORKERS=$(sysctl -n hw.ncpu || echo 2)
else
    NUM_WORKERS=$(nproc || echo 2)
fi

# --- Smart Checkpoint Resuming ---
TRAINING_WEIGHTS="$WEIGHTS"
if [ -f "$OUT/best_model.keras" ]; then
    echo "Found existing local checkpoint under $OUT/best_model.keras. Resuming from there..."
    TRAINING_WEIGHTS="$OUT/best_model.keras"
fi

# --- Weights & Biases Configuration ---
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Warning: WANDB_API_KEY is not set. Logging in anonymous mode..."
    export WANDB_ANONYMOUS="must"
fi

export WANDB_PROJECT="sudan-alpr-ocr"
export WANDB_NOTES="Fine-tuning global cct-xs-v2 model on Sudanese layout splits."
# This environment hook forces wandb to automatically intercept Keras training loops 
export WANDB_KERAS_PATCH="true" 

# --- Execution ---
echo "Starting fine-tuning sequence with WandB sync..."
export KERAS_BACKEND=tensorflow
export TF_CPP_MIN_LOG_LEVEL=2 

# Execute the CLI; WandB patches the underlying Keras `.fit()` callbacks seamlessly
"$PY" train \
  --model-config-file "$CFG/cct_xs_v2_model.yaml" \
  --plate-config-file "$CFG/sudan_plate.yaml" \
  --annotations "$DS/train.csv" \
  --val-annotations "$DS/val.csv" \
  --weights-path "$TRAINING_WEIGHTS" \
  --validate-dataset warn \
  --epochs 120 \
  --batch-size 32 \
  --lr 0.0003 \
  --early-stopping-patience 25 \
  --early-stopping-metric val_plate_acc \
  --output-dir "$OUT" \
  --workers "$NUM_WORKERS"

echo ""
echo "✓ Fine-tuning done. Metrics synced to Weights & Biases dashboard."
