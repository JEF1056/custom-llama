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
    echo "  AutoRound: /models/qwen3.6-27b-autoround  (download first via manage_models.py)"
    echo "  GGUF:      absolute path to a .gguf file inside /models"
    exit 1
fi

# Detect model type: .gguf file on disk vs safetensors directory / HF repo ID
if [[ "$MODEL_PATH" == *.gguf ]]; then
    IS_GGUF=1
    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: model file not found: $MODEL_PATH"
        echo "Run llama-convert to download and quantize a model into /models first."
        exit 1
    fi
else
    IS_GGUF=0
fi

echo "Starting SGLang server"
echo "  Host:     $HOST:$PORT"
echo "  Model:    $MODEL_PATH"
[[ $IS_GGUF -eq 0 ]] && echo "  Format:   safetensors${QUANTIZATION:+ ($QUANTIZATION)}"
[ -n "$TOKENIZER_PATH" ]  && echo "  Tokenizer: $TOKENIZER_PATH"
[ -n "$SERVED_MODEL_NAME" ] && echo "  Name:     $SERVED_MODEL_NAME"
echo "  Context:  $CONTEXT_LENGTH tokens"
echo "  VRAM:     ${MEM_FRACTION} fraction"
echo "  TP:       $TP_SIZE GPU(s)"
[ -n "$REASONING_PARSER" ] && echo "  Reasoning: $REASONING_PARSER"
[ -n "$SPEC_ALGO" ]        && echo "  Speculative: $SPEC_ALGO (steps=$SPEC_NUM_STEPS topk=${SPEC_EAGLE_TOPK:-auto} draft=$SPEC_NUM_DRAFT_TOKENS)"
[ -n "$KV_CACHE_DTYPE" ]   && echo "  KV dtype: $KV_CACHE_DTYPE"
[ -n "$API_KEY" ]          && echo "  API key:  (set)"

# Build flags that differ between GGUF and safetensors models
LOAD_ARGS=()
SPEC_ARGS=()
if [[ $IS_GGUF -eq 1 ]]; then
    LOAD_ARGS+=(--load-format gguf --quantization gguf)
else
    # Apply quantization format for safetensors models (e.g. auto-round, gptq)
    [[ -n "$QUANTIZATION" ]] && LOAD_ARGS+=(--quantization "$QUANTIZATION")
fi

# Speculative decoding applies to both GGUF and safetensors models
if [[ -n "$SPEC_ALGO" ]]; then
    SPEC_ARGS+=(--speculative-algorithm "$SPEC_ALGO"
                --speculative-num-steps "$SPEC_NUM_STEPS"
                --speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS")
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
