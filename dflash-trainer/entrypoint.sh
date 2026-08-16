#!/bin/bash
set -eo pipefail

echo "======================================================="
echo "   DFlash Two-Phase Pipeline for RTX 3090 (24GB VRAM)"
echo "   100% z-lab/Qwen3.6-27B-DFlash Architecture Parity"
echo "======================================================="

MODEL_ID="${MODEL_ID:-/models/qwen38-src/hf_safetensors}"
FEATURES_DIR="${FEATURES_DIR:-/workspace-data/features}"
CACHE_DIR="${CACHE_DIR:-/workspace-data/dataset-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/output/Qwen3.8-27B-heretic-dflash}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
MAX_STEPS="${MAX_STEPS:-10000}"

mkdir -p "$FEATURES_DIR" "$CACHE_DIR" "$OUTPUT_DIR"

if [ ! -f "$FEATURES_DIR/projection_weights.pt" ]; then
    echo ">>> Starting Phase 1: Feature Extraction from Target Model..."
    python3 extract_features.py \
        --model-id "$MODEL_ID" \
        --output-dir "$FEATURES_DIR" \
        --cache-dir "$CACHE_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --seq-len 1024
    echo ">>> Phase 1 Complete!"
else
    echo ">>> Found existing extracted features in $FEATURES_DIR, skipping Phase 1."
fi

echo ">>> Starting Phase 2: Ultra-Fast Dedicated Drafter Training (Zero Spill, Full Speed)..."
python3 train_drafter.py \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --block-size 16 \
    --max-steps "$MAX_STEPS" \
    --lr 2e-4

echo "======================================================="
echo "   DFlash Training Successfully Finished!"
echo "   Exported model in: $OUTPUT_DIR"
echo "======================================================="
