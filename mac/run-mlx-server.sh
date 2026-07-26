#!/usr/bin/env bash
#
# Launches the Qwen3.6-35B-A3B MLX VLM server. Invoked by the LaunchAgent.
#
# Uses mlx_vlm.server for native vision-language model support. The model is
# expected at $MODEL_PATH (default: ~/.qwen/models/qwen36-mlx/quantized).
#
# Prompt caching: Automatic Prefix Caching (APC) is enabled (APC_ENABLED=1) so
# mlx-vlm reuses a shared prefix's KV across requests; on the vision path the
# block hash folds in an image content hash so cached text prefixes stay
# correct across different images. KV cache is 4-bit quantized (--kv-bits 4).
#
# Thinking: --enable-thinking activates the model's native reasoning/thinking
# mode when present in the checkpoint.
#
# No vision toggle needed: mlx_vlm handles image input natively.
set -euo pipefail

# ---- Model path -------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-$HOME/.qwen/models/qwen36-mlx/quantized}

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[qwen36] ERROR: Model directory not found: $MODEL_PATH" >&2
    echo "[qwen36] Run mac/install.sh to download and quantize the model." >&2
    exit 1
fi

# ---- Prompt caching -----------------------------------------------------------
# Export APC_ENABLED so mlx-vlm's apc module picks it up. Set APC_ENABLED=0 to
# disable.
export APC_ENABLED=${APC_ENABLED:-1}

# ---- KV cache quantization ----------------------------------------------------
# 4-bit uniform, the MLX analogue of llama.cpp q4_0. Set MLX_KV_BITS= (empty)
# to keep a full-precision KV cache.
MLX_KV_BITS=${MLX_KV_BITS:-3}

# ---- KV cache size limit ------------------------------------------------------
# Cap the KV cache to prevent OOM on long-context requests. 64K tokens at
# 4-bit ≈ 2-3 GB of KV cache, keeping total memory well within 48 GB.
# Set MLX_MAX_KV_SIZE= (empty) to disable the limit (default: 65536).
MLX_MAX_KV_SIZE=${MLX_MAX_KV_SIZE:-229376}

# ---- Sampling params ----------------------------------------------------------
# Default sampling params (clients may override per request). A modest
# temperature keeps output consistent. Set any to empty to leave the MLX
# server's own default in place.
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}

# Repetition control. repetition_penalty=1.0 is the MLX default; presence_penalty=0.0
# is also the MLX default. Both are forwarded only when explicitly set (non-default).
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}

# ---- Server binding -----------------------------------------------------------
MLX_HOST=${MLX_HOST:-0.0.0.0}
MLX_PORT=${MLX_PORT:-8081}

# ---- Build KV bits argument ---------------------------------------------------
KV_BITS_ARG=""
if [[ -n "$MLX_KV_BITS" ]]; then
    KV_BITS_ARG="--kv-bits $MLX_KV_BITS"
fi

# ---- Build max KV size argument -----------------------------------------------
MAX_KV_SIZE_ARG=""
if [[ -n "$MLX_MAX_KV_SIZE" ]]; then
    MAX_KV_SIZE_ARG="--max-kv-size $MLX_MAX_KV_SIZE"
fi

# ---- Build extra args ---------------------------------------------------------
EXTRA_ARGS=""

# Enable thinking mode if the model supports it (--enable-thinking is a
# mlx_vlm.server flag that activates the model's native chain-of-thought
# / reasoning behavior).
EXTRA_ARGS="$EXTRA_ARGS --enable-thinking --kv-quant-scheme turboquant"

# ---- Build prefill step size argument -----------------------------------------
# Smaller prefill steps reduce peak memory during long-context processing,
# avoiding Metal command buffer OOM (30 GB limit). Default: 1024.
MLX_PREFILL_STEP_SIZE=${MLX_PREFILL_STEP_SIZE:-1024}
PRELOAD_ARGS=""
if [[ -n "$MLX_PREFILL_STEP_SIZE" ]]; then
    PRELOAD_ARGS="--prefill-step-size $MLX_PREFILL_STEP_SIZE"
fi

# ---- Launch the server --------------------------------------------------------
echo "[qwen36] Starting mlx_vlm.server"
echo "[qwen36] Model: $MODEL_PATH"
echo "[qwen36] Host: $MLX_HOST  Port: $MLX_PORT"
echo "[qwen36] KV bits: $MLX_KV_BITS  Max KV size: $MLX_MAX_KV_SIZE  Prefill step: $MLX_PREFILL_STEP_SIZE  APC: $APC_ENABLED"
echo "[qwen36] Sampling defaults: temp=$TEMP top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P"
echo "[qwen36] Repetition penalty: $REPETITION_PENALTY  Presence penalty: $PRESENCE_PENALTY"
echo "[qwen36] Thinking: enabled"

cd "$MODEL_PATH"

python3 -m mlx_vlm.server \
    --host "$MLX_HOST" \
    --port "$MLX_PORT" \
    --model "$MODEL_PATH" \
    $KV_BITS_ARG \
    $MAX_KV_SIZE_ARG \
    $PRELOAD_ARGS \
    $EXTRA_ARGS
