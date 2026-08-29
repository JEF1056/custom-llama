#!/usr/bin/env bash
#
# Container entrypoint for the Qwen3.6-35B-A3B (ik_llama.cpp) CUDA server.
#
# The image contains stock ik_llama.cpp's llama-server. This launches it with:
#   - the in-house "262K-Balanced" quant GGUF (or, for bring-up before that
#     recipe is produced, Unsloth's pre-converted GGUF via -hf/-hff)
#   - 4-bit Hadamard-rotated KV cache on the attention layers (-khad/-vhad)
#   - an optional n-gram lookup drafter chained before MTP as a fast/free
#     first speculative stage (--spec-type ngram-mod:... --spec-type mtp:...),
#     OFF by default - it breaks speculative decoding entirely when the
#     vision tower is also loaded, see ENABLE_NGRAM below
#   - MTP self-speculative decoding (--spec-type mtp:...)
#   - the Qwen3-VL-family vision tower (--mmproj)
#   - prompt/context caching (--cache-ram) and the full 262144 native context
# See docs/iqllama-migration-plan.md for the full design and flag rationale.
set -euo pipefail

log() { printf '\033[1;32m[qwen36-cuda]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[qwen36-cuda]\033[0m %s\n' "$*" >&2; }

PORT=${PORT:-8080}
# Full GPU offload; the 3090 holds the whole quantized model at the sizes in
# docs/iqllama-migration-plan.md section 4c/4d.
NGL=${NGL:-999}
# Flash attention. GPU-only feature; set to off for CPU-only deployments
# (the cpu Compose services set FLASH_ATTN=off automatically).
FLASH_ATTN=${FLASH_ATTN:-on}
# Context window in tokens. Default = 200000 (200k tokens), matching the 3090 profile.
CTX_SIZE=${CTX_SIZE:-${CTX:-200000}}
CTX=${CTX:-$CTX_SIZE}
# KV-cache data type for the 10 full-attention layers (the 30 DeltaNet layers
# use a fixed-size recurrent state, independent of KV type/context length).
# q4_0 is the recipe default (~1.5 GB at 262K, see migration plan section 4b).
KV_TYPE=${KV_TYPE:-q4_0}
KV_TYPE_K=${KV_TYPE_K:-$KV_TYPE}
KV_TYPE_V=${KV_TYPE_V:-$KV_TYPE}
# Hadamard-rotate the K/V cache before quantizing (reduces quant error; see
# migration plan section 0 item 1 / R2). Set 0 to disable.
KV_HADAMARD=${KV_HADAMARD:-1}

# Prompt caching: the server's context-checkpoint + prompt-state cache (full
# sequence-state save/restore). Size its RAM budget in MiB (-1 = no limit,
# 0 = disable).
CACHE_RAM_MIB=${CACHE_RAM_MIB:-1024}
# Balanced checkpoint configuration for DeltaNet recurrent state resumption
# with minimal VRAM overhead: 4 snapshots spaced every 8,192 tokens (~600MB VRAM).
CTX_CHECKPOINTS_N=${CTX_CHECKPOINTS_N:-4}
CTX_CHECKPOINTS_INTERVAL=${CTX_CHECKPOINTS_INTERVAL:-8192}

# Number of concurrent request slots. The server splits the KV context evenly
# across slots, so N_PARALLEL slots cap a single sequence to CTX / N_PARALLEL
# tokens. Keep this at 1 so one request can use the FULL CTX (needed for long
# 100K+ context); raise it only for concurrent serving of shorter sequences.
NP=${NP:-${N_PARALLEL:-1}}
N_PARALLEL=${N_PARALLEL:-$NP}

# Model-loading memory behavior. Since the model is fully GPU-offloaded (NGL=999),
# these mainly affect load-time staging, not steady-state decode throughput.
#   DISABLE_MMAP=1 (--no-mmap): read the whole GGUF into RAM up front instead of
#     mmap'ing it (avoids later page faults if the file cache gets evicted).
#   USE_MLOCK=1 (--mlock): lock resident RAM pages so the OS can never swap them.
#     Needs --ulimit memlock=-1:-1; without it, llama-server may error/warn.
# Both default OFF (mmap is normally faster and lower-RAM).
DISABLE_MMAP=${DISABLE_MMAP:-0}
USE_MLOCK=${USE_MLOCK:-0}

# Physical batch size (-ub): max tokens processed per GPU pass during prompt
# processing. Sized to 512 for low activation VRAM overhead (~500MB buffer vs ~2GB at 1024).
# Physical & logical batch sizes
BATCH_SIZE=${BATCH_SIZE:-4096}
UBATCH_SIZE=${UBATCH_SIZE:-1024}
NGL=${NGL:-99}
NGLD=${NGLD:-99}
THREADS=${THREADS:-16}
THREADS_BATCH=${THREADS_BATCH:-4}

# Vision (multimodal). Qwen3.6-35B-A3B ships a Qwen3-VL-lineage vision tower as
# a separate mmproj GGUF (ik_llama's examples/mtmd/clip.cpp has a full
# PROJECTOR_TYPE_QWEN3VL implementation). Enabled by default; set 0 for text-only.
ENABLE_VISION=${ENABLE_VISION:-0}
MMPROJ_FILE=${MMPROJ_FILE:-mmproj-BF16.gguf}

# DFlash speculative drafter (z-lab/Qwen3.8-27B-DFlash2-GGUF)
ENABLE_DFLASH=${ENABLE_DFLASH:-0}
DFLASH_REPO=${DFLASH_REPO:-z-lab/Qwen3.8-27B-DFlash2-GGUF}
DFLASH_FILE=${DFLASH_FILE:-Qwen3.8-27B-DFlash2-iq4_kss.gguf}
DFLASH_N_MAX=${DFLASH_N_MAX:-3}
DFLASH_P_MIN=${DFLASH_P_MIN:-0.0}

# MTP self-speculative decoding (DeepSeek-V3-style single trailing MTP layer,
# baked into the same GGUF - no separate draft model file needed). Enabled by
# default. MTP_N_MAX = max speculative tokens per round; MTP_P_MIN = minimum
# acceptance probability (0.0 = accept greedily-consistent tokens only).
ENABLE_MTP=${ENABLE_MTP:-1}
if [[ "$ENABLE_DFLASH" == "1" || "$ENABLE_DFLASH" == "true" || "$ENABLE_DFLASH" == "yes" || "$ENABLE_DFLASH" == "on" ]]; then
    ENABLE_MTP=0
fi
MTP_N_MAX=${MTP_N_MAX:-3}
MTP_P_MIN=${MTP_P_MIN:-0.0}
# Optionally requantize the MTP output head independently (e.g. higher-precision
# head raises draft acceptance). Set to iq4_kss for minimal VRAM footprint.
MTP_REQUANTIZE_OUTPUT_TYPE=${MTP_REQUANTIZE_OUTPUT_TYPE:-iq4_kss}

# Two-Stage N-Gram Lookup Drafter: tuned for code decode on single 24GB RTX 3090
if [[ "$ENABLE_VISION" == "1" ]]; then
    ENABLE_NGRAM=0
else
    ENABLE_NGRAM=${ENABLE_NGRAM:-1}
fi
NGRAM_TYPE=${NGRAM_TYPE:-ngram-mod}
NGRAM_N_MAX=${NGRAM_N_MAX:-4}
NGRAM_N_MIN=${NGRAM_N_MIN:-2}
NGRAM_SIZE_N=${NGRAM_SIZE_N:-16}

# Sampling defaults. These set the server-side default generation params;
# clients may still override them per request.
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
REPEAT_PENALTY=${REPEAT_PENALTY:-1.0}

# Reasoning routing (thinking gate)
REASONING=${REASONING:-off}
REASONING_FORMAT=${REASONING_FORMAT:-deepseek}

# Preserve thinking. When on (default), prior assistant turns keep their <think>
# reasoning blocks when the conversation is re-rendered, instead of the template
# stripping them from every turn before the last user query. Set 0 to strip.
REASONING_BUDGET=${REASONING_BUDGET:-4096}
PRESERVE_THINKING=${PRESERVE_THINKING:-1}

# --- Weights -----------------------------------------------------------------
MODEL_SOURCE=${MODEL_SOURCE:-local}
GGUF_FILE=${GGUF_FILE:-qwen38-27b-heretic-ara-iq4_kss.gguf}
HF_REPO=${HF_REPO:-trohrbaugh/Qwen3.8-27B-heretic-ara}
HF_FILE=${HF_FILE:-}
export HF_TOKEN=${QWEN_TOKEN:-${HF_TOKEN:-}}
export LLAMA_CACHE=${LLAMA_CACHE:-/workspace/models}
mkdir -p "$LLAMA_CACHE"

# --- Assemble llama-server flags ---------------------------------------------
ENABLE_JINJA=${ENABLE_JINJA:-1}

SERVER_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    -ngl "$NGL"
    -ngld "$NGLD"
    -fa "$FLASH_ATTN"
    --parallel "$N_PARALLEL"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    -t "$THREADS"
    -tb "$THREADS_BATCH"
    --cache-ram "$CACHE_RAM_MIB"
    --ctx-checkpoints "$CTX_CHECKPOINTS_N"
    --ctx-checkpoints-interval "$CTX_CHECKPOINTS_INTERVAL"
    --cache-type-k "$KV_TYPE_K"
    --cache-type-v "$KV_TYPE_V"
    --recurrent-ckpt-mode auto
    --merge-qkv
    --parallel-tool-calls
    --reasoning "$REASONING"
    --reasoning-format "$REASONING_FORMAT"
    --temp "$TEMP"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --min-p "$MIN_P"
    --presence-penalty "$PRESENCE_PENALTY"
    --repeat-penalty "$REPEAT_PENALTY"
)
case "${ENABLE_JINJA,,}" in
    0|false|no|off|"") ;;
    *) SERVER_ARGS+=(--jinja) ;;
esac
case "${DISABLE_MMAP,,}" in
    0|false|no|off|"") ;;
    *) SERVER_ARGS+=(--no-mmap) ;;
esac
case "${USE_MLOCK,,}" in
    0|false|no|off|"") ;;
    *) SERVER_ARGS+=(--mlock) ;;
esac
case "${MODEL_SOURCE,,}" in
    hf)
        if [[ -z "$HF_FILE" ]]; then
            err "MODEL_SOURCE=hf requires HF_FILE (exact GGUF filename in $HF_REPO)."
            exit 1
        fi
        log "Model source: Hugging Face $HF_REPO / $HF_FILE"
        SERVER_ARGS+=(-hf "$HF_REPO" -hff "$HF_FILE")
        ;;
    local|*)
        MODEL_PATH="/models/${GGUF_FILE}"
        if [[ ! -f "$MODEL_PATH" ]]; then
            err "MODEL_SOURCE=local but $MODEL_PATH is missing."
            err "Run scripts/quantize.sh (see migration plan Phase 2) to produce it,"
            err "or set MODEL_SOURCE=hf for a public bring-up quant."
            exit 1
        fi
        # Create symlink with unified model name for vllm-router load balancing
        MODEL_NAME=${MODEL_NAME:-qwen3.6-35b}
        MODEL_ALIAS_PATH="/models/${MODEL_NAME}"
        if [[ ! -L "$MODEL_ALIAS_PATH" ]]; then
            ln -sf "$MODEL_PATH" "$MODEL_ALIAS_PATH"
        fi
        log "Model source: local $MODEL_PATH (alias: $MODEL_ALIAS_PATH)"
        SERVER_ARGS+=(-m "$MODEL_ALIAS_PATH")
        ;;
esac
if [[ -n "${CTX:-}" && "${CTX}" != "0" ]]; then
    SERVER_ARGS+=(-c "$CTX")
fi
# Hadamard-rotated KV: generic flags, gated purely on head_dim (Qwen3.6's
# attention head_dim=256 is power-of-2 compatible, see migration plan R2).
case "${KV_HADAMARD,,}" in
    0|false|no|off|"") HADAMARD_ON=0 ;;
    *)                  HADAMARD_ON=1 ;;
esac
if [[ "$HADAMARD_ON" == "1" ]]; then
    SERVER_ARGS+=(-khad -vhad)
fi
# Vision: pin the mmproj explicitly (local file if MODEL_SOURCE=local, else
# resolved from the same HF repo). --no-mmproj keeps text-only servers from
# loading/downloading it at all.
case "${ENABLE_VISION,,}" in
    0|false|no|off|"") VISION_ON=0 ;;
    *)                 VISION_ON=1 ;;
esac
if [[ "$VISION_ON" == "1" ]]; then
    if [[ "${MODEL_SOURCE,,}" == "hf" ]]; then
        SERVER_ARGS+=(--mmproj-url "https://huggingface.co/${HF_REPO}/resolve/main/${MMPROJ_FILE}")
    else
        MMPROJ_PATH="/models/${MMPROJ_FILE}"
        if [[ ! -f "$MMPROJ_PATH" ]]; then
            err "ENABLE_VISION=1 but $MMPROJ_PATH is missing; set ENABLE_VISION=0"
            err "or place the mmproj GGUF alongside the model."
            exit 1
        fi
        SERVER_ARGS+=(--mmproj "$MMPROJ_PATH")
    fi
fi
# Speculative Decoding Pipeline
# ik_llama.cpp supports two-stage speculative chaining:
#   Stage 1: Self-spec drafter (e.g. ngram-mod) — zero-latency CPU pattern lookup
#   Stage 2: Neural drafter (MTP head or DFlash model) — GPU semantic speculative decode

case "${ENABLE_NGRAM,,}" in
    0|false|no|off|"") NGRAM_ON=0 ;;
    *)                 NGRAM_ON=1 ;;
esac
case "${ENABLE_DFLASH,,}" in
    1|true|yes|on) DFLASH_ON=1 ;;
    *)             DFLASH_ON=0 ;;
esac
case "${ENABLE_MTP,,}" in
    0|false|no|off|"") MTP_ON=0 ;;
    *)                 MTP_ON=1 ;;
esac

# 1. Load DFlash draft weights if enabled
if [[ "$DFLASH_ON" == "1" ]]; then
    DFLASH_PATH="/models/${DFLASH_FILE}"
    if [[ ! -f "$DFLASH_PATH" ]]; then
        log "Downloading DFlash drafter from https://huggingface.co/${DFLASH_REPO}/resolve/main/${DFLASH_FILE}..."
        mkdir -p /models
        curl -fL "https://huggingface.co/${DFLASH_REPO}/resolve/main/${DFLASH_FILE}" -o "$DFLASH_PATH" || {
            err "Failed to download DFlash drafter from ${DFLASH_REPO}/${DFLASH_FILE}"
            exit 1
        }
    fi
    SERVER_ARGS+=(--model-draft "$DFLASH_PATH")
fi

# 2. Stage 1: Self-spec drafter (must be registered first)
if [[ "$NGRAM_ON" == "1" ]]; then
    SERVER_ARGS+=(--spec-type "${NGRAM_TYPE}:n_max=${NGRAM_N_MAX},n_min=${NGRAM_N_MIN},ngram_size_n=${NGRAM_SIZE_N}")
fi

# 3. Stage 2: Neural drafter (DFlash draft model or native MTP head)
if [[ "$DFLASH_ON" == "1" ]]; then
    SERVER_ARGS+=(--spec-type "draft:n_max=${DFLASH_N_MAX},p_min=${DFLASH_P_MIN}")
elif [[ "$MTP_ON" == "1" ]]; then
    SERVER_ARGS+=(--spec-type "mtp:n_max=${MTP_N_MAX},p_min=${MTP_P_MIN}")
    if [[ -n "$MTP_REQUANTIZE_OUTPUT_TYPE" ]]; then
        SERVER_ARGS+=(--mtp-requantize-output-tensor "$MTP_REQUANTIZE_OUTPUT_TYPE")
    fi
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

log "Starting llama-server on :$PORT | model=${MODEL_SOURCE} ctx=$CTX kv=${KV_TYPE_K}/${KV_TYPE_V} hadamard=$([[ "$HADAMARD_ON" == "1" ]] && echo on || echo off) slots=${N_PARALLEL} preserve-thinking=$([[ "$PRESERVE_THINKING_ON" == "1" ]] && echo on || echo off) cache-ram=${CACHE_RAM_MIB}MiB vision=$([[ "$VISION_ON" == "1" ]] && echo on || echo off) mtp=$([[ "$MTP_ON" == "1" ]] && echo on || echo off) tool-calling=on"
exec llama-server "${SERVER_ARGS[@]}"

