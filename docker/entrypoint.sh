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
# Context window in tokens. Default = the model's full native 262144. Set
# CTX=0 to let ik_llama auto-fit context to available VRAM instead.
CTX=${CTX:-262144}
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
CACHE_RAM_MIB=${CACHE_RAM_MIB:-8192}

# Number of concurrent request slots. The server splits the KV context evenly
# across slots, so N_PARALLEL slots cap a single sequence to CTX / N_PARALLEL
# tokens. Keep this at 1 so one request can use the FULL CTX (needed for long
# 100K+ context); raise it only for concurrent serving of shorter sequences.
N_PARALLEL=${N_PARALLEL:-1}

# Physical batch size (-ub): the max number of tokens processed together in
# one GPU pass during prompt processing. Larger values raise prompt-processing
# throughput (more parallel work per kernel launch) at the cost of more VRAM
# for compute/activation buffers; generation (token-by-token decode) speed is
# essentially unaffected by this value. Empirically swept on a single RTX 3090
# with the ~13GB IQ3_XXS bring-up quant (see docs/qwen36-bench-results.md):
#   ub=256:  ~15.8GB used, ~1818 tok/s prompt processing
#   ub=512:  ~16.1GB used, ~2580 tok/s (ik_llama.cpp default)
#   ub=1024: ~16.6GB used, ~3192 tok/s  <- chosen default: +24% pp throughput
#            over the stock default for only +3% VRAM, leaving ~8GB headroom
#            for a larger production quant.
#   ub=2048: ~17.7GB used, ~3567 tok/s (diminishing returns per extra VRAM)
# Raise toward 2048 if serving a smaller quant with VRAM to spare; lower
# toward 256 if a larger quant leaves little headroom.
UBATCH_SIZE=${UBATCH_SIZE:-1024}

# Vision (multimodal). Qwen3.6-35B-A3B ships a Qwen3-VL-lineage vision tower as
# a separate mmproj GGUF (ik_llama's examples/mtmd/clip.cpp has a complete
# PROJECTOR_TYPE_QWEN3VL implementation). Enabled by default; set
# ENABLE_VISION=0 for a leaner text-only server that skips the mmproj entirely.
ENABLE_VISION=${ENABLE_VISION:-1}
MMPROJ_FILE=${MMPROJ_FILE:-mmproj-BF16.gguf}

# MTP self-speculative decoding (DeepSeek-V3-style single trailing MTP layer,
# baked into the same GGUF - no separate draft model file needed). Enabled by
# default; set ENABLE_MTP=0 to disable. MTP_N_MAX is the max number of
# speculative tokens proposed per round; MTP_P_MIN is the minimum acceptance
# probability threshold (0.0 = accept greedily-consistent tokens only).
ENABLE_MTP=${ENABLE_MTP:-1}
MTP_N_MAX=${MTP_N_MAX:-4}
MTP_P_MIN=${MTP_P_MIN:-0.0}
# Optionally requantize the MTP output head independently of the main output
# tensor (e.g. a higher-precision head raises draft acceptance). Empty = use
# whatever precision the GGUF already baked in for the MTP head.
MTP_REQUANTIZE_OUTPUT_TYPE=${MTP_REQUANTIZE_OUTPUT_TYPE:-}

# Optional n-gram lookup drafter, chained as a first (fast/free) speculative
# stage ahead of MTP (ik_llama.cpp's --spec-type supports a two-stage chain,
# e.g. `--spec-type ngram-mod:... --spec-type mtp:...`). It costs no extra
# model inference - just string matching against the existing context/cache -
# so it's a pure win whenever the response contains repeated or templated
# spans (long-document summarization, code, boilerplate) that a plain n-gram
# match can propose for free, leaving MTP to handle the harder/novel spans.
#
# **ROOT-CAUSED, not just observed (this build, examples/server/server-
# context.cpp)**: when `mmproj` is loaded, that file explicitly whitelists
# ONLY two speculative configs: zero stages, or exactly one stage of type
# MTP (`spec_stages.empty() || (spec_stages.size()==1 &&
# spec_stages.front().type==COMMON_SPECULATIVE_TYPE_MTP)`). Anything else -
# including our ngram-mod+mtp 2-stage chain - fails that check, and the
# `else` branch doesn't just skip the extra stage: it clears ALL stages
# (`stages.clear(); has_mtp=false;`), which is why MTP itself stopped
# working too, not just ngram. This is a deliberate upstream restriction
# (likely because the mtmd/vision-embeddings codepath has only been wired
# up against single-stage MTP, which already needs special embeddings
# handling - see `llama_set_embeddings` right below this check in that
# file), not something fixable from this entrypoint/Dockerfile. Default is
# therefore OFF (ENABLE_NGRAM=0) so vision+MTP keeps working out of the
# box. Only set ENABLE_NGRAM=1 on a text-only deployment (ENABLE_VISION=0) -
# and note
# ENABLE_VISION=0 currently hits ITS OWN pre-existing bug in this build
# (`error: unknown argument: --no-mmproj`), so verify that path separately
# before relying on the ngram+MTP combo.
ENABLE_NGRAM=${ENABLE_NGRAM:-0}
NGRAM_TYPE=${NGRAM_TYPE:-ngram-mod}
NGRAM_N_MAX=${NGRAM_N_MAX:-64}
NGRAM_N_MIN=${NGRAM_N_MIN:-2}
NGRAM_SIZE_N=${NGRAM_SIZE_N:-8}

# Sampling defaults. These set the server-side default generation params;
# clients may still override them per request.
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
# Two ways to source the model, controlled by MODEL_SOURCE:
#   local (default) - our in-house "262K-Balanced" GGUF (docs/
#                      iqllama-migration-plan.md Phase 2/4), produced offline by
#                      scripts/quantize.sh and mounted read-only under /models.
#   hf              - pull a public GGUF straight from Hugging Face via -hf/-hff
#                      (e.g. Unsloth's pre-converted quant) - useful for bring-up
#                      / smoke-testing the engine before our recipe is ready.
MODEL_SOURCE=${MODEL_SOURCE:-local}
GGUF_FILE=${GGUF_FILE:-qwen36-262k-balanced.gguf}
HF_REPO=${HF_REPO:-unsloth/Qwen3.6-35B-A3B-MTP-GGUF}
HF_FILE=${HF_FILE:-}
export HF_TOKEN=${QWEN_TOKEN:-${HF_TOKEN:-}}
# Persist downloaded weights so restarts never re-download. Under docker compose
# this is set to the dedicated /models cache volume; the /workspace fallback
# keeps standalone `docker run` invocations persistent too.
export LLAMA_CACHE=${LLAMA_CACHE:-/workspace/models}
mkdir -p "$LLAMA_CACHE"

# --- Assemble llama-server flags ---------------------------------------------
SERVER_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    -ngl "$NGL"
    -fa on
    --jinja
    --parallel "$N_PARALLEL"
    -ub "$UBATCH_SIZE"
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
            err "Run scripts/quantize.sh (see docs/iqllama-migration-plan.md Phase 2)"
            err "to produce it, or set MODEL_SOURCE=hf for a public bring-up quant."
            exit 1
        fi
        log "Model source: local $MODEL_PATH"
        SERVER_ARGS+=(-m "$MODEL_PATH")
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
            err "or place the mmproj GGUF alongside the model (see Phase 2)."
            exit 1
        fi
        SERVER_ARGS+=(--mmproj "$MMPROJ_PATH")
    fi
else
    SERVER_ARGS+=(--no-mmproj)
fi
# MTP self-speculative decoding: no separate draft file - the trailing MTP
# layer(s) are baked into the same GGUF. If the n-gram drafter is also
# enabled, its --spec-type must be registered FIRST so it forms the first
# (cheap) stage of the two-stage chain, with MTP as the second stage.
case "${ENABLE_MTP,,}" in
    0|false|no|off|"") MTP_ON=0 ;;
    *)                 MTP_ON=1 ;;
esac
case "${ENABLE_NGRAM,,}" in
    0|false|no|off|"") NGRAM_ON=0 ;;
    *)                  NGRAM_ON=1 ;;
esac
if [[ "$NGRAM_ON" == "1" ]]; then
    SERVER_ARGS+=(--spec-type "${NGRAM_TYPE}:n_max=${NGRAM_N_MAX},n_min=${NGRAM_N_MIN},ngram_size_n=${NGRAM_SIZE_N}")
fi
if [[ "$MTP_ON" == "1" ]]; then
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

