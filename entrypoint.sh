#!/bin/bash
set -e

# vLLM server entrypoint.
# All configuration is driven by VLLM_* environment variables.
# Pass extra flags directly: docker compose run vllm-server --log-requests

HOST=${VLLM_HOST:-0.0.0.0}
PORT=${VLLM_PORT:-8080}
MODEL_PATH=${VLLM_MODEL_PATH:-}
TOKENIZER_PATH=${VLLM_TOKENIZER_PATH:-}
SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME:-}
API_KEY=${VLLM_API_KEY:-}
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-128000}
GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1}
TP_SIZE=${VLLM_TP_SIZE:-1}
DTYPE=${VLLM_DTYPE:-auto}
KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-auto}
QUANTIZATION=${VLLM_QUANTIZATION:-}
MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-}
ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-}
REASONING_PARSER=${VLLM_REASONING_PARSER:-}
ENABLE_REASONING=${VLLM_ENABLE_REASONING:-}
SPECULATIVE_CONFIG=${VLLM_SPECULATIVE_CONFIG:-}
ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-}
SWAP_SPACE=${VLLM_SWAP_SPACE:-4}
SCHEDULING_POLICY=${VLLM_SCHEDULING_POLICY:-}
MAX_SEQ_LEN_TO_CAPTURE=${VLLM_MAX_SEQ_LEN_TO_CAPTURE:-}
DISABLE_LOG_REQUESTS=${VLLM_DISABLE_LOG_REQUESTS:-}
ENABLE_METRICS=${VLLM_ENABLE_METRICS:-}

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: VLLM_MODEL_PATH is required"
    echo "  Download a model first: docker compose run --rm model-prep download <model>"
    echo "  Then set: VLLM_MODEL_PATH=/models/<model-name>"
    exit 1
fi

echo "Starting vLLM server"
echo "  Host:     $HOST:$PORT"
echo "  Model:    $MODEL_PATH"
echo "  Format:   safetensors${QUANTIZATION:+ ($QUANTIZATION)}"
[ -n "$TOKENIZER_PATH" ]    && echo "  Tokenizer: $TOKENIZER_PATH"
[ -n "$SERVED_MODEL_NAME" ] && echo "  Name:     $SERVED_MODEL_NAME"
echo "  Context:  $MAX_MODEL_LEN tokens"
echo "  GPU mem:  ${GPU_MEMORY_UTILIZATION} utilization"
echo "  TP:       $TP_SIZE GPU(s)"
echo "  Seqs:     $MAX_NUM_SEQS concurrent"
[ -n "$REASONING_PARSER" ]  && echo "  Reasoning: $REASONING_PARSER"
[ -n "$SPECULATIVE_CONFIG" ] && echo "  Speculative: $SPECULATIVE_CONFIG"
[ "$KV_CACHE_DTYPE" != "auto" ] && echo "  KV dtype: $KV_CACHE_DTYPE"
[ -n "$ENFORCE_EAGER" ]     && echo "  CUDA graph: disabled (enforce eager)"
[ -n "$API_KEY" ]           && echo "  API key:  (set)"

exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --swap-space "$SWAP_SPACE" \
    --trust-remote-code \
    ${TOKENIZER_PATH:+--tokenizer "$TOKENIZER_PATH"} \
    ${SERVED_MODEL_NAME:+--served-model-name "$SERVED_MODEL_NAME"} \
    ${API_KEY:+--api-key "$API_KEY"} \
    ${QUANTIZATION:+--quantization "$QUANTIZATION"} \
    ${MAX_NUM_SEQS:+--max-num-seqs "$MAX_NUM_SEQS"} \
    ${MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"} \
    ${KV_CACHE_DTYPE:+--kv-cache-dtype "$KV_CACHE_DTYPE"} \
    ${ENABLE_CHUNKED_PREFILL:+--enable-chunked-prefill} \
    ${ENABLE_PREFIX_CACHING:+--enable-prefix-caching} \
    ${ENABLE_REASONING:+--enable-reasoning --reasoning-parser "$REASONING_PARSER"} \
    ${SPECULATIVE_CONFIG:+--speculative-config "$SPECULATIVE_CONFIG"} \
    ${ENFORCE_EAGER:+--enforce-eager} \
    ${SCHEDULING_POLICY:+--scheduling-policy "$SCHEDULING_POLICY"} \
    ${MAX_SEQ_LEN_TO_CAPTURE:+--max-seq-len-to-capture "$MAX_SEQ_LEN_TO_CAPTURE"} \
    ${DISABLE_LOG_REQUESTS:+--disable-log-requests} \
    ${ENABLE_METRICS:+--enable-metrics} \
    "$@"
