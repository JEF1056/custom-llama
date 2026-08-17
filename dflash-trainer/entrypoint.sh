#!/usr/bin/env bash
set -e

echo "======================================================="
echo "   DFlash Two-Phase Pipeline for RTX 3090 (24GB VRAM)"
echo "   Multi-Language & Reasoning Training"
echo "======================================================="

MODEL_ID=${MODEL_ID:-"/models/qwen38-src/hf_safetensors"}
FEATURES_DIR=${FEATURES_DIR:-"/workspace-data/features"}
CACHE_DIR=${CACHE_DIR:-"/workspace-data/dataset-cache"}
OUTPUT_DIR=${OUTPUT_DIR:-"/output/Qwen3.8-27B-heretic-dflash"}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
CHUNK_SIZE=${CHUNK_SIZE:-500}
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_STEPS=${MAX_STEPS:-10000}

mkdir -p "$FEATURES_DIR" "$OUTPUT_DIR" "$CACHE_DIR"

SHARD_COUNT=$(find "$FEATURES_DIR" -name "shard_*.pt" 2>/dev/null | wc -l)
PROJ_EXISTS=$(test -f "$FEATURES_DIR/projection_weights.pt" && echo "yes" || echo "no")

if [ "$SHARD_COUNT" -ge 20 ] && [ "$PROJ_EXISTS" = "yes" ]; then
    echo ">>> Found complete Phase 1 features ($SHARD_COUNT/20 shards + projection_weights.pt)."
    echo ">>> Bypassing Phase 1 and jumping directly to Phase 2 training!"
else
    echo ">>> Phase 1 incomplete or missing ($SHARD_COUNT/20 shards found)."
    echo ">>> Starting Phase 1: Multi-Language Feature Extraction from Target Model ($NUM_SAMPLES samples)..."
    python3 extract_features.py \
        --model-id "$MODEL_ID" \
        --output-dir "$FEATURES_DIR" \
        --cache-dir "$CACHE_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --chunk-size "$CHUNK_SIZE" \
        --seq-len 1024
    echo ">>> Phase 1 Complete!"
fi

echo ">>> Starting Phase 2: Ultra-Fast Dedicated Drafter Training (Batch Size: $BATCH_SIZE, Block Size: 4, Multi-Worker, 10k Steps)..."
python3 train_drafter.py \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --block-size 4 \
    --max-steps "$MAX_STEPS" \
    --lr 8e-4

echo "======================================================="
echo "   DFlash Training Successfully Finished!"
echo "   Exported model in: $OUTPUT_DIR"
echo "======================================================="
