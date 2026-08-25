#!/usr/bin/env bash
set -e

echo "======================================================="
echo "   DFlash 2 Two-Phase Pipeline for RTX 3090 (24GB VRAM)"
echo "   Multi-Language & Reasoning Training with Fixed Benchmark"
echo "======================================================="

MODEL_ID=${MODEL_ID:-"/models/qwen38-src/hf_safetensors"}
FEATURES_DIR=${FEATURES_DIR:-"/workspace-data/features"}
FIXED_VAL_DIR=${FIXED_VAL_DIR:-"/workspace-data/fixed_val_features"}
CACHE_DIR=${CACHE_DIR:-"/workspace-data/dataset-cache"}
OUTPUT_DIR=${OUTPUT_DIR:-"/output/Qwen3.8-27B-heretic-dflash2"}
NUM_SAMPLES=${NUM_SAMPLES:-3000}
SKIP_SAMPLES=${SKIP_SAMPLES:-0}
CHUNK_SIZE=${CHUNK_SIZE:-150}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
CTX_LEN=${CTX_LEN:-2048}
SEQ_LEN=${SEQ_LEN:-2064}
BLOCK_SIZE=${BLOCK_SIZE:-8}
LR=${LR:-"5e-5"}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MAX_STEPS=${MAX_STEPS:-3000}
PRETRAINED_DRAFTER=${PRETRAINED_DRAFTER:-"z-lab/Qwen3.8-27B-DFlash2"}

export TORCH_COMPILE_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=4

mkdir -p "$FEATURES_DIR" "$FIXED_VAL_DIR" "$OUTPUT_DIR" "$CACHE_DIR"

EXPECTED_SHARDS=$(( NUM_SAMPLES / CHUNK_SIZE ))
SHARD_COUNT=$(find "$FEATURES_DIR" -name "shard_*.pt*" 2>/dev/null | wc -l)
PROJ_EXISTS=$(test -f "$FEATURES_DIR/embed.safetensors" && echo "yes" || echo "no")

if [ "$SHARD_COUNT" -ge "$EXPECTED_SHARDS" ] && [ "$PROJ_EXISTS" = "yes" ]; then
    echo ">>> Found complete Phase 1 features ($SHARD_COUNT/$EXPECTED_SHARDS shards + embed.safetensors)."
    echo ">>> Bypassing Phase 1 and jumping directly to Phase 2 training!"
else
    echo ">>> Phase 1 incomplete or missing ($SHARD_COUNT/$EXPECTED_SHARDS shards found)."
    echo ">>> Starting Phase 1: Feature Extraction ($NUM_SAMPLES samples, Skip: $SKIP_SAMPLES, Seq Len: $SEQ_LEN, Chunk Size: $CHUNK_SIZE)..."
    python3 extract_features.py \
        --model-id "$MODEL_ID" \
        --output-dir "$FEATURES_DIR" \
        --fixed-val-dir "$FIXED_VAL_DIR" \
        --cache-dir "$CACHE_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --skip-samples "$SKIP_SAMPLES" \
        --chunk-size "$CHUNK_SIZE" \
        --seq-len "$SEQ_LEN"
    echo ">>> Phase 1 Complete!"
fi

echo ">>> Starting Phase 2: DFlash 2 Drafter Training from $PRETRAINED_DRAFTER (Batch Size: $BATCH_SIZE x $GRAD_ACCUM_STEPS, Ctx Len: $CTX_LEN, Block Size: $BLOCK_SIZE, LR: $LR, $MAX_STEPS Steps)..."
python3 train_drafter.py \
    --features-dir "$FEATURES_DIR" \
    --fixed-val-dir "$FIXED_VAL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --ctx-len "$CTX_LEN" \
    --block-size "$BLOCK_SIZE" \
    --max-steps "$MAX_STEPS" \
    --lr "$LR" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --pretrained-drafter "$PRETRAINED_DRAFTER"

echo "======================================================="
echo "   DFlash 2 Training Successfully Finished!"
echo "   Exported model in: $OUTPUT_DIR"
echo "======================================================="
