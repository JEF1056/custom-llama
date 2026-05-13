#!/bin/bash
set -e

# Default values
HOST=${LLAMA_HOST:-0.0.0.0}
PORT=${LLAMA_PORT:-8080}
THREADS=${LLAMA_THREADS:-6}
THREADS_BATCH=${LLAMA_THREADS_BATCH:-4}
CTX_SIZE=${LLAMA_CTX_SIZE:-200000}
BATCH_SIZE=${LLAMA_BATCH_SIZE:-4096}
UBATCH_SIZE=${LLAMA_UBATCH_SIZE:-2048}
GPU_LAYERS=${LLAMA_GPU_LAYERS:-99}
MAX_TOKENS=${LLAMA_MAX_TOKENS:--1}
TOP_P=${LLAMA_TOP_P:-0.95}
TEMP=${LLAMA_TEMP:-0.6}
TOP_K=${LLAMA_TOP_K:-0}
MIN_P=${LLAMA_MIN_P:-0.0}
PRESENCE_PENALTY=${LLAMA_PRESENCE_PENALTY:-0.0}
REPETITION_PENALTY=${LLAMA_REPETITION_PENALTY:-1.0}
STOP=${LLAMA_STOP:-}
DRY_MULTIPLIER=${LLAMA_DRY_MULTIPLIER:-0}
DRY_BASE=${LLAMA_DRY_BASE:-1.75}
DRY_ALLOWED_LENGTH=${LLAMA_DRY_ALLOWED_LENGTH:-2}
DRY_PENALTY_LAST_N=${LLAMA_DRY_PENALTY_LAST_N:-512}
REASONING_BUDGET=${LLAMA_REASONING_BUDGET:-0}

# TurboQuant KV-cache settings
# K cache type: f16 (default), turbo3 (TurboQuant 3-bit), turbo4 (TurboQuant 4-bit)
# V cache type: f16 (default), turbo3 (TurboQuant 3-bit), turbo4 (TurboQuant 4-bit)
CACHE_TYPE_K=${LLAMA_CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${LLAMA_CACHE_TYPE_V:-f16}

# Flash Attention: required for TurboQuant KV cache (auto-enabled by server, but explicit is better)
FLASH_ATTN=${LLAMA_FLASH_ATTN:-on}

# Parallel inference slots (concurrent requests)
# Does not increase VRAM usage — slots share the same model weights, only KV cache is duplicated per slot
PARALLEL=${LLAMA_PARALLEL:-2}

# Memory mapping (off = mmap enabled, which is efficient for large models)
# Set to "on" to disable mmap (loads entire model into RAM first — faster but requires more RAM)
NO_MMAP=${LLAMA_NO_MMAP:-off}

# Direct I/O: bypass OS page cache when loading the model file.
# Recommended when all layers are GPU-offloaded — avoids caching ~9 GB of model
# weights in RAM that are already resident in VRAM.
DIRECT_IO=${LLAMA_DIRECT_IO:-off}

# Reasoning mode (on = chain-of-thought output for reasoning models)
# Per-model setting; enable for models trained with reasoning capabilities (e.g., DeepSeek-R1, QwQ)
REASONING=${LLAMA_REASONING:-on}

# preserve_thinking: inject prior <think>…</think> blocks back into the prompt each turn.
# Qwopus3.6 / Qwen3.6 chat templates support this kwarg natively via preserve_thinking=true.
PRESERVE_THINKING=${LLAMA_PRESERVE_THINKING:-on}


# KV offload: off = allow llama.cpp to spill KV to CPU if VRAM runs low (safe default).
#             on  = pass --no-kv-offload, forcing KV cache to stay on GPU at all times.
NO_KV_OFFLOAD=${LLAMA_NO_KV_OFFLOAD:-off}

# Unified KV cache across slots: on = --kv-unified (share full context pool across all slots).
# Default in llama.cpp ≥ b4550; explicit here for clarity.
KV_UNIFIED=${LLAMA_KV_UNIFIED:-on}

# Host-memory prompt cache size in MiB (--cache-ram).
# -1 = unlimited, 0 = disabled, empty = omit flag (let llama.cpp default apply).
CACHE_RAM=${LLAMA_CACHE_RAM:-}

# Clear idle slots to host RAM on each new task (--clear-idle).
# Requires CACHE_RAM to be set and non-zero. Reduces n_kv to active tokens only.
CLEAR_IDLE=${LLAMA_CLEAR_IDLE:-on}

# Tensor split for multi-GPU (e.g., "13,14" splits model across GPU 0 and GPU 1)
# Leave empty for single GPU — model must fit on one GPU
TS=${LLAMA_TS:-}

# MoE CPU offload (number of experts to offload to CPU)
# For Mixture-of-Experts models like Mixtral — controls which experts run on CPU vs GPU
NCMOE=${LLAMA_NCMOE:-}

# KV cache slot save path — directory where slot state is written to disk via
# POST /slots/{id}?action=save and read back via ?action=restore.
# Enables per-conversation KV cache persistence across server restarts.
# Requires llama.cpp >= b3000 (upstream) or equivalent TurboQuant build.
SLOT_SAVE_PATH=${LLAMA_SLOT_SAVE_PATH:-}

# API key — when set, all requests to the server must include
# Authorization: Bearer <key>. Leave empty for unauthenticated access
# (appropriate when protected by Tailscale ACLs).
API_KEY=${LLAMA_API_KEY:-}

# Multimodal settings
MMPROJ=${LLAMA_MMPROJ:-}  # Multimodal projector file (.mmproj)
IMAGE=${LLAMA_IMAGE:-}     # Image file for multimodal input (base64 or path)

# Auto-discover mmproj when MODEL_NAME is set and LLAMA_MMPROJ is not explicit.
# manage_models.py downloads mmproj as {model_name}-mmproj.gguf, so we can
# locate it without requiring the user to set LLAMA_MMPROJ manually.
if [ -z "$MMPROJ" ] && [ -n "$MODEL_NAME" ]; then
    _auto_mmproj="/models/${MODEL_NAME}-mmproj.gguf"
    if [ -f "$_auto_mmproj" ]; then
        MMPROJ="$_auto_mmproj"
    fi
    unset _auto_mmproj
fi

# Determine model path.
# Priority:
#   1) LLAMA_MODEL — explicit path to a .gguf already in /models
#   2) MODEL_NAME + QUANT — constructs /models/{MODEL_NAME}-{QUANT}.gguf
#      (model must have been prepared beforehand by the convert image)
#   3) fallback to /models/model.gguf
if [ -n "$LLAMA_MODEL" ]; then
    MODEL="$LLAMA_MODEL"
elif [ -n "$MODEL_NAME" ]; then
    QUANT="${QUANT:-${TQ_QUANT:-IQ4_XS}}"
    MODEL="/models/${MODEL_NAME}-${QUANT}.gguf"
else
    MODEL=/models/model.gguf
fi

# Verify the model file exists — models must be prepared in advance using the
# convert image (docker compose run --rm llama-convert download/convert-st …).
# Create slot save directory if a path is configured.
if [ -n "$SLOT_SAVE_PATH" ]; then
    mkdir -p "$SLOT_SAVE_PATH"
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    echo ""
    echo "Prepare the model first using the convert image, for example:"
    if [ -n "$MODEL_NAME" ]; then
        echo "  docker compose run --rm llama-convert download $MODEL_NAME --quant ${QUANT:-IQ4_XS}"
        echo "  # or for safetensors-only models:"
        echo "  docker compose run --rm llama-convert convert-st $MODEL_NAME --quant ${QUANT:-TQ2_0}"
    else
        echo "  docker compose run --rm llama-convert download <model-name> --quant <quant>"
    fi
    exit 1
fi
echo "Model file found: $MODEL"

# Check if multimodal projector exists (if specified)
if [ -n "$MMPROJ" ] && [ ! -f "$MMPROJ" ]; then
    echo "WARNING: Multimodal projector file not found: $MMPROJ"
    MMPROJ=""
fi

echo "Starting llama-server with configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL"
echo "  Threads (decode): $THREADS"
echo "  Threads (batch):  $THREADS_BATCH"
echo "  Context Size: $CTX_SIZE"
echo "  Batch Size: $BATCH_SIZE  (ubatch: $UBATCH_SIZE)"
echo "  GPU Layers: $GPU_LAYERS"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Top P: $TOP_P"
echo "  Temperature: $TEMP"
if [ -n "$STOP" ]; then
    echo "  Stop Sequences: $STOP"
fi
if [ -n "$MMPROJ" ]; then
    echo "  Multimodal Projector: $MMPROJ"
fi
if [ -n "$IMAGE" ]; then
    echo "  Image: $IMAGE"
fi
echo "  KV Cache K: $CACHE_TYPE_K"
echo "  KV Cache V: $CACHE_TYPE_V"
echo "  Flash Attention: $FLASH_ATTN"
echo "  Parallel Slots: $PARALLEL"
echo "  Memory Mapping: $([ "$NO_MMAP" = "on" ] && echo "disabled (no-mmap)" || echo "enabled (mmap)")"
echo "  Direct I/O: $([ "$DIRECT_IO" = "on" ] && echo "enabled" || echo "disabled")"
echo "  Reasoning Mode: $REASONING"
echo "  Preserve Thinking: $PRESERVE_THINKING"
echo "  No KV Offload: $([ "$NO_KV_OFFLOAD" = "on" ] && echo "enabled" || echo "disabled")"
echo "  KV Unified:    $([ "$KV_UNIFIED" = "on" ] && echo "enabled" || echo "disabled")"
if [ -n "$CACHE_RAM" ]; then
    echo "  Cache RAM:     $([ "$CACHE_RAM" = "-1" ] && echo "unlimited" || echo "${CACHE_RAM} MiB")"
fi
if [ -n "$CACHE_RAM" ] && [ "$CACHE_RAM" != "0" ]; then
    echo "  Clear Idle:    $([ "$CLEAR_IDLE" = "on" ] && echo "enabled" || echo "disabled")"
fi
if [ -n "$TS" ]; then
    echo "  Tensor Split: $TS"
fi
if [ -n "$NCMOE" ]; then
    echo "  MoE CPU Offload: $NCMOE"
fi
if [ -n "$SLOT_SAVE_PATH" ]; then
    echo "  Slot Save Path: $SLOT_SAVE_PATH"
fi
if [ -n "$API_KEY" ]; then
    echo "  API Key: (set)"
fi

# Build multimodal flags
MMFLAGS=""
if [ -n "$MMPROJ" ]; then
    MMFLAGS="$MMFLAGS --mmproj $MMPROJ"
fi
if [ -n "$IMAGE" ]; then
    MMFLAGS="$MMFLAGS --image $IMAGE"
fi

# Execute llama-server with arguments
exec llama-server \
    --webui-mcp-proxy \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --n-gpu-layers "$GPU_LAYERS" \
    -n "$MAX_TOKENS" \
    --top-p "$TOP_P" \
    --temp "$TEMP" \
    ${TOP_K:+--top-k "$TOP_K"} \
    ${MIN_P:+--min-p "$MIN_P"} \
    ${PRESENCE_PENALTY:+--presence-penalty "$PRESENCE_PENALTY"} \
    ${REPETITION_PENALTY:+--repeat-penalty "$REPETITION_PENALTY"} \
    --dry-multiplier "$DRY_MULTIPLIER" \
    --dry-base "$DRY_BASE" \
    --dry-allowed-length "$DRY_ALLOWED_LENGTH" \
    --dry-penalty-last-n "$DRY_PENALTY_LAST_N" \
    $([ "$REASONING_BUDGET" -gt 0 ] 2>/dev/null && echo "--reasoning-budget $REASONING_BUDGET") \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --flash-attn "$FLASH_ATTN" \
    ${STOP:+--stop "$STOP"} \
    ${PARALLEL:+--parallel "$PARALLEL"} \
    $([ "$NO_MMAP" = "on" ] && echo "--no-mmap") \
    $([ "$DIRECT_IO" = "on" ] && echo "--direct-io") \
    --reasoning "$REASONING" \
    $([ "$PRESERVE_THINKING" = "on" ] && echo '--chat-template-kwargs {"preserve_thinking":true}') \
    $([ "$NO_KV_OFFLOAD" = "on" ] && echo "--no-kv-offload") \
    $([ "$KV_UNIFIED" = "on" ] && echo "--kv-unified") \
    ${CACHE_RAM:+--cache-ram "$CACHE_RAM"} \
    $([ "$CLEAR_IDLE" = "on" ] && [ -n "$CACHE_RAM" ] && [ "$CACHE_RAM" != "0" ] && echo "--clear-idle") \
    $([ "$CLEAR_IDLE" = "off" ] && [ -n "$CACHE_RAM" ] && [ "$CACHE_RAM" != "0" ] && echo "--no-clear-idle") \
    ${TS:+--tensor-split "$TS"} \
    ${NCMOE:+-ncmoe "$NCMOE"} \
    ${SLOT_SAVE_PATH:+--slot-save-path "$SLOT_SAVE_PATH"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    $MMFLAGS \
    "$@"
