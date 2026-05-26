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
SPEC_ALGO=${SGLANG_SPECULATIVE_ALGO:-}
SPEC_NUM_STEPS=${SGLANG_SPECULATIVE_NUM_STEPS:-3}
SPEC_EAGLE_TOPK=${SGLANG_SPECULATIVE_EAGLE_TOPK:-}
SPEC_NUM_DRAFT_TOKENS=${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: SGLANG_MODEL_PATH is required"
    echo "Set SGLANG_MODEL_PATH to the absolute path of a GGUF file inside /models"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model file not found: $MODEL_PATH"
    echo "Run llama-convert to download and quantize a model into /models first."
    exit 1
fi

echo "Starting SGLang server"
echo "  Host:     $HOST:$PORT"
echo "  Model:    $MODEL_PATH"
[ -n "$TOKENIZER_PATH" ]  && echo "  Tokenizer: $TOKENIZER_PATH"
[ -n "$SERVED_MODEL_NAME" ] && echo "  Name:     $SERVED_MODEL_NAME"
echo "  Context:  $CONTEXT_LENGTH tokens"
echo "  VRAM:     ${MEM_FRACTION} fraction"
echo "  TP:       $TP_SIZE GPU(s)"
[ -n "$REASONING_PARSER" ] && echo "  Reasoning: $REASONING_PARSER"
[ -n "$SPEC_ALGO" ]        && echo "  Speculative: $SPEC_ALGO (steps=$SPEC_NUM_STEPS topk=${SPEC_EAGLE_TOPK:-auto} draft=$SPEC_NUM_DRAFT_TOKENS)"
[ -n "$KV_CACHE_DTYPE" ]   && echo "  KV dtype: $KV_CACHE_DTYPE"
[ -n "$API_KEY" ]          && echo "  API key:  (set)"

exec python3.12 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --load-format gguf \
    --quantization gguf \
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
    ${SPEC_ALGO:+--speculative-algorithm "$SPEC_ALGO"} \
    ${SPEC_ALGO:+--speculative-num-steps "$SPEC_NUM_STEPS"} \
    ${SPEC_ALGO:+--speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS"} \
    ${SPEC_EAGLE_TOPK:+--speculative-eagle-topk "$SPEC_EAGLE_TOPK"} \
    "$@"
