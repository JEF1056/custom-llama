#!/usr/bin/env bash
#
# Container entrypoint for the Bonsai-27B (1-bit) CUDA server.
#
# The image already contains llama-server built from the TurboQuant+ fork, so
# this just pulls the 1-bit GGUF weights from Hugging Face (cached in the mounted
# volume) and launches the server with the full 262K context, tool calling,
# prompt/prefix caching, DSpark speculative decoding and (optionally) the Bonsai
# vision tower.
#
# DSpark and prompt/prefix caching run together: both ride the same RS-rollback +
# checkpoint machinery, so there is no speculative-vs-cache trade-off here. The
# DSpark drafter is a separate GGUF loaded as the draft model (ENABLE_DSPARK=1 by
# default; set ENABLE_DSPARK=0 for a plain non-speculative server).
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
# KV_TYPE sets both sides at once; KV_TYPE_K / KV_TYPE_V override each side
# independently (they default to KV_TYPE) for asymmetric K/V, e.g. turbo4 keys
# + turbo2 values.
KV_TYPE=${KV_TYPE:-q4_0}
KV_TYPE_K=${KV_TYPE_K:-$KV_TYPE}
KV_TYPE_V=${KV_TYPE_V:-$KV_TYPE}
# Prompt caching. Bonsai is a HYBRID (GDN + attention) arch whose recurrent
# state can't be position-shifted, so llama.cpp auto-disables --cache-reuse (KV
# shifting) on it. The mechanism that DOES work here is the server's context
# checkpoints + prompt-state cache (full sequence-state save/restore) - on by
# default, and the same machinery speculative decoding reuses. Size its RAM
# budget in MiB (-1 = no limit, 0 = disable).
CACHE_RAM_MIB=${CACHE_RAM_MIB:-8192}

# Number of concurrent request slots. The server splits the KV context evenly
# across slots, so N_PARALLEL slots cap a single sequence to CTX / N_PARALLEL
# tokens. Keep this at 1 so one request can use the FULL CTX (needed for long
# 100K+ context); raise it only for concurrent serving of shorter sequences.
N_PARALLEL=${N_PARALLEL:-1}

# Vision (multimodal). Bonsai ships a ~0.46B vision tower as a separate mmproj
# GGUF; the fork loads it through the existing Qwen3-VL projector, so image
# input works with no code changes. Enabled by default; set ENABLE_VISION=0 for
# a leaner text-only server that skips the ~629 MB mmproj download. MMPROJ_FILE
# selects the pack - the Q8_0 container holds the HQQ 4-bit tower (~629 MB);
# Bonsai-27B-mmproj-BF16.gguf (~931 MB) is the higher-precision reference.
ENABLE_VISION=${ENABLE_VISION:-1}
MMPROJ_FILE=${MMPROJ_FILE:-Bonsai-27B-mmproj-Q8_0.gguf}

# DSpark speculative decoding. The Bonsai DSpark drafter is a separate ~1.79 GB
# GGUF (Q4_1) served as the draft model via --spec-type draft-dspark. It shares
# the RS-rollback + checkpoint path with prompt caching, so both run together.
# Enabled by default; set ENABLE_DSPARK=0 for a plain non-speculative server.
ENABLE_DSPARK=${ENABLE_DSPARK:-1}
DSPARK_DRAFT_FILE=${DSPARK_DRAFT_FILE:-Bonsai-27B-dspark-Q4_1.gguf}
# Sizes the server's decode output reserve as n_parallel * (1 + this). Must be
# >= the drafter checkpoint's block_size (4 for Bonsai-27B) for the block draft.
# NOTE: on this fork the dspark capture pass currently flags ~every prompt token
# as an output row, so the reserve really needs to reach n_batch; but sizing it
# that high (via a large value here) makes context creation OOM on a 24 GB GPU.
# There is no config value that both loads and survives long prompts -- this
# needs a fork-side fix (capture should not force n_outputs_max == n_batch).
DSPARK_N_MAX=${DSPARK_N_MAX:-8}

# Sampling defaults. These set the server-side default generation params;
# clients may still override them per request. Lower temperature raises DSpark
# draft acceptance (the drafter proposes near-greedily), so keep it modest.
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
REPEAT_PENALTY=${REPEAT_PENALTY:-1.0}

# Preserve thinking. When on (default), prior assistant turns keep their <think>
# reasoning blocks when the conversation is re-rendered, instead of the template
# stripping them from every turn before the last user query. Set 0 to strip.
PRESERVE_THINKING=${PRESERVE_THINKING:-1}

# --- Weights -----------------------------------------------------------------
# The prism-ml Bonsai-27B GGUF repos are public, so no token is needed for the
# default weights; a read token is only required for gated/private repos.
# BONSAI_TOKEN wins, else HF_TOKEN.
HF_REPO=${HF_REPO:-prism-ml/Bonsai-27B-gguf}
HF_FILE=${HF_FILE:-Bonsai-27B-Q1_0.gguf}
export HF_TOKEN=${BONSAI_TOKEN:-${HF_TOKEN:-}}
# Persist downloaded weights so restarts never re-download. Under docker compose
# this is set to the dedicated /models cache volume; the /workspace fallback
# keeps standalone `docker run` invocations persistent too.
export LLAMA_CACHE=${LLAMA_CACHE:-/workspace/models}
mkdir -p "$LLAMA_CACHE"

if [[ -z "${HF_TOKEN:-}" ]]; then
    log "No BONSAI_TOKEN/HF_TOKEN set - downloading anonymously (the default"
    log "prism-ml Bonsai-27B GGUF repos are public). Set a token only if you"
    log "point HF_REPO at a gated or private repo."
fi

# --- Assemble llama-server flags ---------------------------------------------
SERVER_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    -hf "${HF_REPO}"
    -hff "${HF_FILE}"
    -ngl "$NGL"
    -fa on
    --jinja
    --parallel "$N_PARALLEL"
    --cache-ram "$CACHE_RAM_MIB"
    --cache-type-k "$KV_TYPE_K"
    --cache-type-v "$KV_TYPE_V"
    --temp "$TEMP"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --min-p "$MIN_P"
    --presence-penalty "$PRESENCE_PENALTY"
    --repeat-penalty "$REPEAT_PENALTY"
)
if [[ -n "${CTX:-}" && "${CTX}" != "0" ]]; then
    SERVER_ARGS+=(-c "$CTX")
fi
# Vision: pin the mmproj explicitly. The URL download reuses the HF token, and
# giving an explicit file avoids the ambiguous auto-pick between the two mmproj
# packs; --no-mmproj keeps text-only servers from fetching it at all.
case "${ENABLE_VISION,,}" in
    0|false|no|off|"") VISION_ON=0 ;;
    *)                 VISION_ON=1 ;;
esac
if [[ "$VISION_ON" == "1" ]]; then
    SERVER_ARGS+=(--mmproj-url "https://huggingface.co/${HF_REPO}/resolve/main/${MMPROJ_FILE}")
else
    SERVER_ARGS+=(--no-mmproj)
fi
# DSpark speculative decoding: pull the drafter GGUF as the draft model (the
# download reuses HF_TOKEN) and select the block-diffusion draft type. Prompt
# caching stays on - they share the same checkpoint/RS path.
case "${ENABLE_DSPARK,,}" in
    0|false|no|off|"") DSPARK_ON=0 ;;
    *)                 DSPARK_ON=1 ;;
esac
if [[ "$DSPARK_ON" == "1" ]]; then
    # -hfd only takes repo[:quant] (there is no exact-file flag for the draft),
    # and :quant is matched case-insensitively, so derive the quant tag from the
    # drafter filename (e.g. Bonsai-27B-dspark-Q4_1.gguf -> Q4_1).
    DSPARK_DRAFT_QUANT="${DSPARK_DRAFT_FILE##*-}"
    DSPARK_DRAFT_QUANT="${DSPARK_DRAFT_QUANT%.gguf}"
    SERVER_ARGS+=(
        --spec-type draft-dspark
        -hfd "${HF_REPO}:${DSPARK_DRAFT_QUANT}"
        --spec-draft-n-max "${DSPARK_N_MAX}"
    )
fi
# Cap thinking length for clients that don't request a reasoning effort.
if [[ -n "${REASONING_BUDGET:-}" ]]; then
    SERVER_ARGS+=(--reasoning-budget "$REASONING_BUDGET")
fi
# Preserve prior-turn <think> blocks when re-rendering the chat (template kwarg).
case "${PRESERVE_THINKING,,}" in
    0|false|no|off|"") PRESERVE_THINKING_ON=0 ;;
    *)                 PRESERVE_THINKING_ON=1 ;;
esac
if [[ "$PRESERVE_THINKING_ON" == "1" ]]; then
    SERVER_ARGS+=(--chat-template-kwargs '{"preserve_thinking": true}')
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

log "Starting llama-server on :$PORT | model=$HF_FILE ctx=$CTX kv=${KV_TYPE_K}/${KV_TYPE_V} slots=${N_PARALLEL} preserve-thinking=$([[ "$PRESERVE_THINKING_ON" == "1" ]] && echo on || echo off) cache-ram=${CACHE_RAM_MIB}MiB vision=$([[ "$VISION_ON" == "1" ]] && echo on || echo off) dspark=$([[ "$DSPARK_ON" == "1" ]] && echo on || echo off) tool-calling=on"
exec llama-server "${SERVER_ARGS[@]}"
