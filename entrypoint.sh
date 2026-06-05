#!/bin/bash
set -e

# vLLM server entrypoint.
# All configuration is driven by LLM_* environment variables.
# Pass extra flags directly: docker compose run vllm-server --log-requests

HOST=${LLM_HOST:-0.0.0.0}
PORT=${LLM_PORT:-8080}
MODEL_PATH=${LLM_MODEL_PATH:-}
TOKENIZER_PATH=${LLM_TOKENIZER_PATH:-}
SERVED_MODEL_NAME=${LLM_SERVED_MODEL_NAME:-}
API_KEY=${LLM_API_KEY:-}
MAX_MODEL_LEN=${LLM_MAX_MODEL_LEN:-128000}
GPU_MEMORY_UTILIZATION=${LLM_GPU_MEMORY_UTILIZATION:-0.95}
MAX_NUM_SEQS=${LLM_MAX_NUM_SEQS:-1}
TP_SIZE=${LLM_TP_SIZE:-1}
DTYPE=${LLM_DTYPE:-auto}
KV_CACHE_DTYPE=${LLM_KV_CACHE_DTYPE:-auto}
QUANTIZATION=${LLM_QUANTIZATION:-}
MAX_NUM_BATCHED_TOKENS=${LLM_MAX_NUM_BATCHED_TOKENS:-}
ENABLE_CHUNKED_PREFILL=${LLM_ENABLE_CHUNKED_PREFILL:-}
CHUNKED_PREFILL_SIZE=${LLM_CHUNKED_PREFILL_SIZE:-}
ENABLE_PREFIX_CACHING=${LLM_ENABLE_PREFIX_CACHING:-}
REASONING_PARSER=${LLM_REASONING_PARSER:-}
REASONING_CONFIG=${LLM_REASONING_CONFIG:-}
TOOL_CALL_PARSER=${LLM_TOOL_CALL_PARSER:-}
SPECULATIVE_CONFIG=${LLM_SPECULATIVE_CONFIG:-}
ENFORCE_EAGER=${LLM_ENFORCE_EAGER:-}
SCHEDULING_POLICY=${LLM_SCHEDULING_POLICY:-}
DISABLE_LOG_REQUESTS=${LLM_DISABLE_LOG_REQUESTS:-}
CPU_OFFLOAD_GB=${LLM_CPU_OFFLOAD_GB:-}
OPTIMIZATION_LEVEL=${LLM_OPTIMIZATION_LEVEL:-}
MAX_NUM_PARTIAL_PREFILLS=${LLM_MAX_NUM_PARTIAL_PREFILLS:-}
LONG_PREFILL_TOKEN_THRESHOLD=${LLM_LONG_PREFILL_TOKEN_THRESHOLD:-}
GENERATION_CONFIG=${LLM_GENERATION_CONFIG:-}
OVERRIDE_GENERATION_CONFIG=${LLM_OVERRIDE_GENERATION_CONFIG:-}
DRY_CONFIG=${LLM_DRY_CONFIG:-}

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: LLM_MODEL_PATH is required"
    echo "  Download a model first: docker compose run --rm model-prep download <model>"
    echo "  Then set: LLM_MODEL_PATH=/models/<model-name>"
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
[ -n "$REASONING_CONFIG" ]  && echo "  Reasoning config: $REASONING_CONFIG"
[ -n "$SPECULATIVE_CONFIG" ] && echo "  Speculative: $SPECULATIVE_CONFIG"
[ "$KV_CACHE_DTYPE" != "auto" ] && echo "  KV dtype: $KV_CACHE_DTYPE"
[ -n "$ENFORCE_EAGER" ]     && echo "  CUDA graph: disabled (enforce eager)"
[ -n "$CPU_OFFLOAD_GB" ] && [ "$CPU_OFFLOAD_GB" != "0" ] && echo "  CPU offload: ${CPU_OFFLOAD_GB} GB"
[ -n "$OPTIMIZATION_LEVEL" ] && echo "  Optimization: -O$OPTIMIZATION_LEVEL"
[ -n "$OVERRIDE_GENERATION_CONFIG" ] && echo "  Gen config: $OVERRIDE_GENERATION_CONFIG"
[ -n "$DRY_CONFIG" ]        && echo "  DRY config: $DRY_CONFIG"
[ -n "$API_KEY" ]           && echo "  API key:  (set)"

# Clear stale torch.compile cache from previous runs.
# Code changes (fork overlay, new image) invalidate cached AOT artifacts;
# stale entries cause copy_misaligned_inputs crashes at startup.
if [ -d "/root/.cache/vllm/torch_compile_cache" ]; then
    echo "  Clearing stale torch.compile cache"
    rm -rf /root/.cache/vllm/torch_compile_cache
fi

# Prevent numba/OpenMP segfault: numba's default 'omp' backend (libgomp)
# conflicts with PyTorch's libomp in the same process.
export NUMBA_THREADING_LAYER=workqueue

exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
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
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    ${REASONING_CONFIG:+--reasoning-config "$REASONING_CONFIG"} \
    ${TOOL_CALL_PARSER:+--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER"} \
    ${SPECULATIVE_CONFIG:+--speculative-config "$SPECULATIVE_CONFIG"} \
    ${ENFORCE_EAGER:+--enforce-eager} \
    ${SCHEDULING_POLICY:+--scheduling-policy "$SCHEDULING_POLICY"} \
    ${DISABLE_LOG_REQUESTS:+--disable-log-requests} \
    ${CPU_OFFLOAD_GB:+--cpu-offload-gb "$CPU_OFFLOAD_GB"} \
    ${OPTIMIZATION_LEVEL:+-O "$OPTIMIZATION_LEVEL"} \
    ${MAX_NUM_PARTIAL_PREFILLS:+--max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS"} \
    ${LONG_PREFILL_TOKEN_THRESHOLD:+--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD"} \
    ${GENERATION_CONFIG:+--generation-config "$GENERATION_CONFIG"} \
    ${OVERRIDE_GENERATION_CONFIG:+--override-generation-config "$OVERRIDE_GENERATION_CONFIG"} \
    ${DRY_CONFIG:+--dry-config "$DRY_CONFIG"} \
    "$@"
