#!/bin/bash
set -e

# Default values
HOST=${LLAMA_HOST:-0.0.0.0}
PORT=${LLAMA_PORT:-8080}
THREADS=${LLAMA_THREADS:-6}
THREADS_BATCH=${LLAMA_THREADS_BATCH:-12}
CTX_SIZE=${LLAMA_CTX_SIZE:-200000}
BATCH_SIZE=${LLAMA_BATCH_SIZE:-4096}
UBATCH_SIZE=${LLAMA_UBATCH_SIZE:-1024}
GPU_LAYERS=${LLAMA_GPU_LAYERS:-99}
MAX_TOKENS=${LLAMA_MAX_TOKENS:--1}
TOP_P=${LLAMA_TOP_P:-0.95}
TEMP=${LLAMA_TEMP:-0.7}
STOP=${LLAMA_STOP:-}

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

# Reasoning mode (on = chain-of-thought output for reasoning models)
# Per-model setting; enable for models trained with reasoning capabilities (e.g., DeepSeek-R1, QwQ)
REASONING=${LLAMA_REASONING:-on}

# preserve_thinking: inject prior <think>…</think> blocks back into the prompt each turn.
# Qwopus3.6 / Qwen3.6 chat templates support this kwarg natively via preserve_thinking=true.
PRESERVE_THINKING=${LLAMA_PRESERVE_THINKING:-on}


# KV offload: off = allow llama.cpp to spill KV to CPU if VRAM runs low (safe default).
#             on  = pass --no-kv-offload, forcing KV cache to stay on GPU at all times.
NO_KV_OFFLOAD=${LLAMA_NO_KV_OFFLOAD:-off}

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

# Determine model path.
# Priority:
#   1) LLAMA_MODEL — explicit path to a .gguf already in /models
#   2) MODEL_NAME + QUANT — constructs /models/{MODEL_NAME}-{QUANT}.gguf
#      (model must have been prepared beforehand by the convert image)
#   3) fallback to /models/model.gguf
if [ -n "$LLAMA_MODEL" ]; then
    MODEL="$LLAMA_MODEL"
elif [ -n "$MODEL_NAME" ]; then
    QUANT="${QUANT:-${TQ_QUANT:-Q4_K_M}}"
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
        echo "  docker compose run --rm llama-convert download $MODEL_NAME --quant ${QUANT:-Q4_K_M}"
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
echo "  Reasoning Mode: $REASONING"
echo "  Preserve Thinking: $PRESERVE_THINKING"
echo "  No KV Offload: $([ "$NO_KV_OFFLOAD" = "on" ] && echo "enabled" || echo "disabled")"
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
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --flash-attn "$FLASH_ATTN" \
    ${STOP:+--stop "$STOP"} \
    ${PARALLEL:+--parallel "$PARALLEL"} \
    $([ "$NO_MMAP" = "on" ] && echo "--no-mmap") \
    --reasoning "$REASONING" \
    $([ "$PRESERVE_THINKING" = "on" ] && echo '--chat-template-kwargs {"preserve_thinking":true}') \
    $([ "$NO_KV_OFFLOAD" = "on" ] && echo "--no-kv-offload") \
    ${TS:+--tensor-split "$TS"} \
    ${NCMOE:+-ncmoe "$NCMOE"} \
    ${SLOT_SAVE_PATH:+--slot-save-path "$SLOT_SAVE_PATH"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    $MMFLAGS \
    "$@"
