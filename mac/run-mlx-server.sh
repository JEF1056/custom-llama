#!/usr/bin/env bash
#
# Launches the Qwen3.8-27B-heretic-ara MLX VLM server with DFlash Speculative Decoding.
set -euo pipefail

# Ensure standard Homebrew / local python paths are available
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.qwen/mlx-venv/bin:$PATH"

# Resolve python binary
if [[ -x "/opt/homebrew/bin/python3" ]]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
elif [[ -x "$HOME/.qwen/mlx-venv/bin/python3" ]]; then
    PYTHON_BIN="$HOME/.qwen/mlx-venv/bin/python3"
else
    PYTHON_BIN="$(which python3)"
fi

# ---- Model path -------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"$HOME/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4"}
DRAFT_MODEL=${DRAFT_MODEL:-"jfan/Qwen3.8-27B-heretic-dflash"}

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[qwen38] ERROR: Model directory not found: $MODEL_PATH" >&2
    exit 1
fi

# ---- Server binding -----------------------------------------------------------
MLX_HOST=${MLX_HOST:-"0.0.0.0"}
MLX_PORT=${MLX_PORT:-"8080"}

# ---- Tuning parameters --------------------------------------------------------
KV_BITS=${KV_BITS:-"4"}
KV_QUANT_SCHEME=${KV_QUANT_SCHEME:-"uniform"}
KV_GROUP_SIZE=${KV_GROUP_SIZE:-"64"}
MAX_KV_SIZE=${MAX_KV_SIZE:-"131072"}
PREFILL_STEP_SIZE=${PREFILL_STEP_SIZE:-"2048"}
DRAFT_NUM_TOKENS=${DRAFT_NUM_TOKENS:-"3"}

# ---- APC (Automatic Prefix Caching) -------------------------------------------
export APC_ENABLED=1
export APC_NUM_BLOCKS=16384
export APC_BLOCK_SIZE=16
export APC_EXACT_CACHE_ENTRIES=16
export APC_DISK_PATH="$HOME/.cache/mlx-vlm/caching"
export APC_DISK_MAX_GB=40
export APC_DISK_SHARD_MAX_BLOCKS=1024
export APC_DISK_WORKERS=4
export MLX_METAL_FAST_SYNCHRONIZATION=1
mkdir -p "$APC_DISK_PATH"

echo "[qwen38] Starting mlx_vlm.server with $PYTHON_BIN"
echo "[qwen38] Model: $MODEL_PATH"
echo "[qwen38] Draft Model: $DRAFT_MODEL (Tokens: $DRAFT_NUM_TOKENS)"
echo "[qwen38] Host: $MLX_HOST  Port: $MLX_PORT"
echo "[qwen38] KV bits: $KV_BITS ($KV_QUANT_SCHEME)  Max KV size: $MAX_KV_SIZE  Prefill step: $PREFILL_STEP_SIZE"

cd "$MODEL_PATH"

exec "$PYTHON_BIN" -m mlx_vlm.server \
    --host "$MLX_HOST" \
    --port "$MLX_PORT" \
    --model "$MODEL_PATH" \
    --draft-model "$DRAFT_MODEL" \
    --draft-kind dflash \
    --draft-num-tokens "$DRAFT_NUM_TOKENS" \
    --trust-remote-code \
    --kv-bits "$KV_BITS" \
    --kv-quant-scheme "$KV_QUANT_SCHEME" \
    --kv-group-size "$KV_GROUP_SIZE" \
    --prefill-step-size "$PREFILL_STEP_SIZE" \
    --max-kv-size "$MAX_KV_SIZE"
