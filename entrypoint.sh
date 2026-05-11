#!/bin/bash
set -e

# Default values
HOST=${LLAMA_HOST:-0.0.0.0}
PORT=${LLAMA_PORT:-8080}
THREADS=${LLAMA_THREADS:-8}
CTX_SIZE=${LLAMA_CTX_SIZE:-4096}
GPU_LAYERS=${LLAMA_GPU_LAYERS:-99}
MAX_TOKENS=${LLAMA_MAX_TOKENS:-512}
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

# Prompt caching capacity (in tokens)
CACHE_CAPACITY=${LLAMA_CACHE_CAPACITY:-4096}

# Multimodal optimizations
# Multi-KV attention: reduces KV cache size for multimodal models by sharing KV heads across image tokens
MUL_KV=${LLAMA_MUL_KV:-on}
# Cache chunk size for multimodal: controls how KV cache chunks are allocated for image tokens
CACHE_CHUNK_SIZE=${LLAMA_CACHE_CHUNK_SIZE:-0}
# No KV offload: keeps KV cache on GPU for multimodal (faster access for image tokens)
NO_KV_OFFLOAD=${LLAMA_NO_KV_OFFLOAD:-off}

# Tensor split for multi-GPU (e.g., "13,14" splits model across GPU 0 and GPU 1)
# Leave empty for single GPU — model must fit on one GPU
TS=${LLAMA_TS:-}

# MoE CPU offload (number of experts to offload to CPU)
# For Mixture-of-Experts models like Mixtral — controls which experts run on CPU vs GPU
NCMOE=${LLAMA_NCMOE:-}

# Multimodal settings
MMPROJ=${LLAMA_MMPROJ:-}  # Multimodal projector file (.mmproj)
IMAGE=${LLAMA_IMAGE:-}     # Image file for multimodal input (base64 or path)

# Determine model path
# Priority: 1) LLAMA_MODEL env var, 2) default model.gguf
if [ -n "$LLAMA_MODEL" ]; then
    MODEL="$LLAMA_MODEL"
else
    MODEL=/models/model.gguf
fi

# Download model at runtime if MODEL_NAME is set
if [ -n "$MODEL_NAME" ]; then
    echo "========================================"
    echo "  Model Download"
    echo "========================================"
    echo "Model: $MODEL_NAME"
    echo "Quantization: ${MODEL_QUANT:-Q4_K_M}"
    echo "TurboQuant: ${TQ_QUANT:-none}"
    echo ""
    
    # Download the model
    echo "Downloading model..."
    python /scripts/manage_models.py download "$MODEL_NAME" \
        -q "${MODEL_QUANT:-Q4_K_M}" \
        -o /models
    
    # Determine the downloaded model file
    MODEL_FILE="/models/${MODEL_NAME}-${MODEL_QUANT:-Q4_K_M}.gguf"
    
    # Convert to TurboQuant if requested
    if [ -n "$TQ_QUANT" ]; then
        echo ""
        echo "Converting to TurboQuant ($TQ_QUANT)..."
        TQ_MODEL="/models/${MODEL_NAME}-${MODEL_QUANT:-Q4_K_M}-${TQ_QUANT}.gguf"
        llama-quantize "$MODEL_FILE" "$TQ_MODEL" "$TQ_QUANT"
        MODEL="$TQ_MODEL"
        echo "TurboQuant model: $TQ_MODEL"
    else
        MODEL="$MODEL_FILE"
        echo "Base model: $MODEL_FILE"
    fi
    
    # Download mmproj if requested
    if [ -n "$MMPROJ" ]; then
        echo ""
        echo "Downloading multimodal projector..."
        python /scripts/manage_models.py download "$MODEL_NAME" \
            -q "${MODEL_QUANT:-Q4_K_M}" \
            -o /models
        MMPROJ="/models/${MODEL_NAME}-${MODEL_QUANT:-Q4_K_M}-mmproj.gguf"
        echo "Multimodal projector: $MMPROJ"
    fi
    
    echo ""
    echo "Model download complete!"
    echo ""
fi

# Wait for model file to exist (handles model-manager download race condition)
echo "Waiting for model file: $MODEL"
MAX_WAIT=600  # 10 minutes max wait
WAIT_COUNT=0
while [ ! -f "$MODEL" ]; do
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "ERROR: Model file not found after ${MAX_WAIT}s. Aborting."
        echo "Please ensure your model is in /models/ or set LLAMA_MODEL environment variable"
        exit 1
    fi
    echo "  Waiting for model file... (${WAIT_COUNT}s/${MAX_WAIT}s)"
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done
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
echo "  Threads: $THREADS"
echo "  Context Size: $CTX_SIZE"
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
echo "  Cache Capacity: $CACHE_CAPACITY"
echo "  Multi-KV Attention: $MUL_KV"
echo "  Cache Chunk Size: $CACHE_CHUNK_SIZE"
echo "  No KV Offload: $([ "$NO_KV_OFFLOAD" = "on" ] && echo "enabled" || echo "disabled")"
if [ -n "$TS" ]; then
    echo "  Tensor Split: $TS"
fi
if [ -n "$NCMOE" ]; then
    echo "  MoE CPU Offload: $NCMOE"
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
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$GPU_LAYERS" \
    --max-tokens "$MAX_TOKENS" \
    --top-p "$TOP_P" \
    --temp "$TEMP" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    -fa "$FLASH_ATTN" \
    ${STOP:+--stop "$STOP"} \
    ${PARALLEL:+--parallel "$PARALLEL"} \
    ${NO_MMAP:+--no-mmap} \
    ${REASONING:+--reasoning "$REASONING"} \
    --cache-capacity "$CACHE_CAPACITY" \
    ${MUL_KV:+--mul-kv} \
    ${CACHE_CHUNK_SIZE:+--cache-chunk-size "$CACHE_CHUNK_SIZE"} \
    ${NO_KV_OFFLOAD:+--no-kv-offload} \
    ${TS:+-ts "$TS"} \
    ${NCMOE:+-ncmoe "$NCMOE"} \
    $MMFLAGS \
    "$@"
