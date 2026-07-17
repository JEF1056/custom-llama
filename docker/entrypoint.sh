#!/usr/bin/env bash
#
# Container entrypoint for the Bonsai-27B (1-bit) CUDA server.
#
# The image already contains llama-server built from the TurboQuant+ fork, so
# this just pulls the 1-bit GGUF weights from Hugging Face (cached in the mounted
# volume) and launches the server with the full 262K context, tool calling and
# prompt/prefix caching.
#
# NOTE: DSpark speculative decoding is not in this build yet (it is being ported
# into the fork). Because there is no DSpark, prompt/prefix caching is always on
# here - there is no speculative-vs-cache trade-off in this image.
set -euo pipefail

log() { printf '\033[1;32m[bonsai-cuda]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[bonsai-cuda]\033[0m %s\n' "$*" >&2; }

PORT=${PORT:-8080}
# Full GPU offload; the 3090 holds the whole 1-bit model.
NGL=${NGL:-999}
# Context window in tokens. Default = the model's full 262K. The quantized KV
# cache keeps even the full window inside the 3090's 24 GB. Set CTX=0 for
# auto-fit.
CTX=${CTX:-262144}
# KV-cache data type. q4_0 is the safe ~4-bit default that fits 262K on 24 GB.
# The fork also offers TurboQuant KV: turbo4 / turbo3 / turbo2 (higher quality
# at similar or smaller size) - set KV_TYPE=turbo4 to use it.
KV_TYPE=${KV_TYPE:-q4_0}
# Prefix-cache reuse window (tokens). Enables cheap multi-turn / agentic reuse.
CACHE_REUSE=${CACHE_REUSE:-256}

# --- Weights -----------------------------------------------------------------
# Private HF repo -> needs a read token. BONSAI_TOKEN wins, else HF_TOKEN.
HF_REPO=${HF_REPO:-prism-ml/Bonsai-27B-gguf}
HF_FILE=${HF_FILE:-Bonsai-27B-Q1_0.gguf}
export HF_TOKEN=${BONSAI_TOKEN:-${HF_TOKEN:-}}
# Persist downloaded weights in the mounted volume so restarts are instant.
export LLAMA_CACHE=${LLAMA_CACHE:-/workspace/models}
mkdir -p "$LLAMA_CACHE"

if [[ -z "${HF_TOKEN:-}" ]]; then
    err "BONSAI_TOKEN/HF_TOKEN is not set. The Bonsai-27B repo is currently"
    err "private; the weight download will fail without a read token."
fi

# --- Assemble llama-server flags ---------------------------------------------
SERVER_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    -hf "${HF_REPO}:${HF_FILE}"
    -ngl "$NGL"
    -fa on
    --jinja
    --cache-reuse "$CACHE_REUSE"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
)
if [[ -n "${CTX:-}" && "${CTX}" != "0" ]]; then
    SERVER_ARGS+=(-c "$CTX")
fi
# Cap thinking length for clients that don't request a reasoning effort.
if [[ -n "${REASONING_BUDGET:-}" ]]; then
    SERVER_ARGS+=(--reasoning-budget "$REASONING_BUDGET")
fi
# Optional prompt-cache slot directory (persist reusable KV slots to disk).
if [[ -n "${SLOT_SAVE_PATH:-}" ]]; then
    mkdir -p "$SLOT_SAVE_PATH"
    SERVER_ARGS+=(--slot-save-path "$SLOT_SAVE_PATH")
fi
# Space-separated extra args for advanced tuning.
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SERVER_ARGS+=(${EXTRA_ARGS})
fi

log "Starting llama-server on :$PORT | model=$HF_FILE ctx=$CTX kv=$KV_TYPE cache-reuse=$CACHE_REUSE tool-calling=on"
exec llama-server "${SERVER_ARGS[@]}"
