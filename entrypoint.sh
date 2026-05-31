#!/bin/bash
set -e

# SGLang server entrypoint.
# All configuration is driven by SGLANG_* environment variables.
# Pass extra flags directly: docker compose run sglang-server --log-requests

HOST=${SGLANG_HOST:-0.0.0.0}
PORT=${SGLANG_PORT:-8080}
MODEL_PATH=${SGLANG_MODEL_PATH:-}
TOKENIZER_PATH=${SGLANG_TOKENIZER_PATH:-}
SERVED_MODEL_NAME=${SGLANG_SERVED_MODEL_NAME:-}
API_KEY=${SGLANG_API_KEY:-}
CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-131072}
MEM_FRACTION=${SGLANG_MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-}
TP_SIZE=${SGLANG_TP_SIZE:-1}
DTYPE=${SGLANG_DTYPE:-auto}
KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-}
CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-}
REASONING_PARSER=${SGLANG_REASONING_PARSER:-}
QUANTIZATION=${SGLANG_QUANTIZATION:-}
SPEC_ALGO=${SGLANG_SPECULATIVE_ALGO:-}
SPEC_NUM_STEPS=${SGLANG_SPECULATIVE_NUM_STEPS:-3}
SPEC_EAGLE_TOPK=${SGLANG_SPECULATIVE_EAGLE_TOPK:-}
SPEC_NUM_DRAFT_TOKENS=${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: SGLANG_MODEL_PATH is required"
    echo "  Download a model first: docker compose run --rm model-prep download qwen3.6-27b-autoround"
    echo "  Then set: SGLANG_MODEL_PATH=/models/qwen3.6-27b-autoround"
    exit 1
fi

echo "Starting SGLang server"
echo "  Host:     $HOST:$PORT"
echo "  Model:    $MODEL_PATH"
echo "  Format:   safetensors${QUANTIZATION:+ ($QUANTIZATION)}"
[ -n "$TOKENIZER_PATH" ]   && echo "  Tokenizer: $TOKENIZER_PATH"
[ -n "$SERVED_MODEL_NAME" ] && echo "  Name:     $SERVED_MODEL_NAME"
echo "  Context:  $CONTEXT_LENGTH tokens"
echo "  VRAM:     ${MEM_FRACTION} fraction"
echo "  TP:       $TP_SIZE GPU(s)"
[ -n "$REASONING_PARSER" ] && echo "  Reasoning: $REASONING_PARSER"
[ -n "$SPEC_ALGO" ]        && echo "  Speculative: $SPEC_ALGO (steps=$SPEC_NUM_STEPS topk=${SPEC_EAGLE_TOPK:-auto} draft=$SPEC_NUM_DRAFT_TOKENS)"
[ -n "$KV_CACHE_DTYPE" ]   && echo "  KV dtype: $KV_CACHE_DTYPE"
[ -n "$API_KEY" ]          && echo "  API key:  (set)"

LOAD_ARGS=()
SPEC_ARGS=()

[[ -n "$QUANTIZATION" ]] && LOAD_ARGS+=(--quantization "$QUANTIZATION")

if [[ -n "$SPEC_ALGO" ]]; then
    # SGLANG_ENABLE_SPEC_V2 + extra_buffer required for speculative decoding
    # compatibility with radix cache on Qwen3/Mamba-style models.
    export SGLANG_ENABLE_SPEC_V2=1
    SPEC_ARGS+=(--speculative-algorithm "$SPEC_ALGO"
                --speculative-num-steps "$SPEC_NUM_STEPS"
                --speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS"
                --mamba-scheduler-strategy extra_buffer)
    [[ -n "$SPEC_EAGLE_TOPK" ]] && SPEC_ARGS+=(--speculative-eagle-topk "$SPEC_EAGLE_TOPK")
fi

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    "${LOAD_ARGS[@]}" \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION" \
    --dtype "$DTYPE" \
    --tp-size "$TP_SIZE" \
    --trust-remote-code \
    ${TOKENIZER_PATH:+--tokenizer-path "$TOKENIZER_PATH"} \
    ${SERVED_MODEL_NAME:+--served-model-name "$SERVED_MODEL_NAME"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    ${MAX_RUNNING_REQUESTS:+--max-running-requests "$MAX_RUNNING_REQUESTS"} \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    ${CHUNKED_PREFILL_SIZE:+--chunked-prefill-size "$CHUNKED_PREFILL_SIZE"} \
    ${KV_CACHE_DTYPE:+--kv-cache-dtype "$KV_CACHE_DTYPE"} \
    "${SPEC_ARGS[@]}" \
    "$@"
