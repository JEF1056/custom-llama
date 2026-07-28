#!/usr/bin/env bash
#
# Launches the Qwen3.6-35B-A3B MLX VLM server. Invoked by the supervisor.
# All values are hardcoded — no env var dependencies at boot.
set -euo pipefail

# ---- Model path -------------------------------------------------------------
MODEL_PATH="$HOME/.qwen/models/qwen36-mlx"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[qwen36] ERROR: Model directory not found: $MODEL_PATH" >&2
    echo "[qwen36] Run mac/install.sh to download and quantize the model." >&2
    exit 1
fi

# ---- Server binding -----------------------------------------------------------
MLX_HOST="0.0.0.0"
MLX_PORT="8080"

# ---- Build KV bits argument ---------------------------------------------------
KV_BITS_ARG="--kv-bits 4.2"

# ---- Build max KV size argument -----------------------------------------------
MAX_KV_SIZE_ARG="--max-kv-size 229376"

# ---- Build prefill step size argument -----------------------------------------
PRELOAD_ARGS="--prefill-step-size 1024"

# ---- Build extra args ---------------------------------------------------------
EXTRA_ARGS="--enable-thinking --thinking-budget 4096 --kv-quant-scheme turboquant"

# ---- DFlash speculative decoding ----------------------------------------------
DRAFT_MODEL_ARG="--draft-model z-lab/Qwen3.6-35B-A3B-DFlash"

# ---- Launch the server --------------------------------------------------------
echo "[qwen36] Starting mlx_vlm.server"
echo "[qwen36] Model: $MODEL_PATH"
echo "[qwen36] Host: $MLX_HOST  Port: $MLX_PORT"
echo "[qwen36] KV bits: 4.2 (TurboQuant: 4-bit keys, 2-bit values)  Max KV size: 229376  Prefill step: 1024"
echo "[qwen36] Thinking: enabled"
echo "[qwen36] APC: enabled  (blocks=2048  disk=$HOME/.cache/mlx-vlm/caching)"
echo "[qwen36] DFlash: enabled (z-lab/Qwen3.6-35B-A3B-DFlash)"

# ---- APC (Automatic Prefix Caching) -------------------------------------------
export APC_ENABLED=1
export APC_NUM_BLOCKS=2048
export APC_BLOCK_SIZE=16
export APC_DISK_PATH="$HOME/.cache/mlx-vlm/caching"
export APC_DISK_MAX_GB=0
export APC_DISK_SHARD_MAX_BLOCKS=256
export APC_MAX_POOL_TENSORS=450000
export APC_LAYER_MAJOR_MEMORY_MIN_TOKENS=50000
export APC_HASH=fast
mkdir -p "$APC_DISK_PATH"

cd "$MODEL_PATH"

python3 -m mlx_vlm.server \
    --host "$MLX_HOST" \
    --port "$MLX_PORT" \
    --model "$MODEL_PATH" \
    $DRAFT_MODEL_ARG \
    $KV_BITS_ARG \
    $MAX_KV_SIZE_ARG \
    $PRELOAD_ARGS \
    $EXTRA_ARGS
