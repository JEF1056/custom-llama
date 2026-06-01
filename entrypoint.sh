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
CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-145000}
MEM_FRACTION=${SGLANG_MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-}
TP_SIZE=${SGLANG_TP_SIZE:-1}
DTYPE=${SGLANG_DTYPE:-float16}
KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-}
CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-}
MAX_QUEUED_REQUESTS=${SGLANG_MAX_QUEUED_REQUESTS:-}
REASONING_PARSER=${SGLANG_REASONING_PARSER:-}
QUANTIZATION=${SGLANG_QUANTIZATION:-}
SCHEDULE_POLICY=${SGLANG_SCHEDULE_POLICY:-fcfs}
SAMPLING_DEFAULTS=${SGLANG_SAMPLING_DEFAULTS:-}
NUM_CONTINUOUS_DECODE_STEPS=${SGLANG_NUM_CONTINUOUS_DECODE_STEPS:-5}
ENABLE_MIXED_CHUNK=${SGLANG_ENABLE_MIXED_CHUNK:-}
ENABLE_MULTIMODAL=${SGLANG_ENABLE_MULTIMODAL:-}
ENABLE_PIECEWISE_CUDA_GRAPH=${SGLANG_ENABLE_PIECEWISE_CUDA_GRAPH:-}
MAMBA_SCHEDULER_STRATEGY=${SGLANG_MAMBA_SCHEDULER_STRATEGY:-extra_buffer}
MAMBA_SSM_DTYPE=${SGLANG_MAMBA_SSM_DTYPE:-}
MAMBA_FULL_MEMORY_RATIO=${SGLANG_MAMBA_FULL_MEMORY_RATIO:-}
MAX_MAMBA_CACHE_SIZE=${SGLANG_MAX_MAMBA_CACHE_SIZE:-}
LOG_LEVEL=${SGLANG_LOG_LEVEL:-info}
LOG_REQUESTS=${SGLANG_LOG_REQUESTS:-}
ENABLE_METRICS=${SGLANG_ENABLE_METRICS:-}
SPEC_ALGO=${SGLANG_SPECULATIVE_ALGO:-}
SPEC_NUM_STEPS=${SGLANG_SPECULATIVE_NUM_STEPS:-3}
SPEC_EAGLE_TOPK=${SGLANG_SPECULATIVE_EAGLE_TOPK:-}
SPEC_NUM_DRAFT_TOKENS=${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-3}

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
[ -n "$ENABLE_MULTIMODAL" ] && echo "  Vision:   enabled"
[ -n "$ENABLE_PIECEWISE_CUDA_GRAPH" ] && echo "  Piecewise CUDA graph: enabled"
echo "  Mamba:    strategy=$MAMBA_SCHEDULER_STRATEGY${MAMBA_SSM_DTYPE:+ ssm_dtype=$MAMBA_SSM_DTYPE}${MAMBA_FULL_MEMORY_RATIO:+ full_mem_ratio=$MAMBA_FULL_MEMORY_RATIO}"
[ -n "$API_KEY" ]          && echo "  API key:  (set)"

LOAD_ARGS=()
SPEC_ARGS=()

[[ -n "$QUANTIZATION" ]] && LOAD_ARGS+=(--quantization "$QUANTIZATION")

if [[ -n "$SPEC_ALGO" ]]; then
    # SGLANG_ENABLE_SPEC_V2 required for overlap scheduling on EAGLE/NEXTN.
    # extra_buffer mamba strategy required when spec decoding + radix cache are both active.
    export SGLANG_ENABLE_SPEC_V2=1
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
    ${MAX_QUEUED_REQUESTS:+--max-queued-requests "$MAX_QUEUED_REQUESTS"} \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    ${CHUNKED_PREFILL_SIZE:+--chunked-prefill-size "$CHUNKED_PREFILL_SIZE"} \
    ${KV_CACHE_DTYPE:+--kv-cache-dtype "$KV_CACHE_DTYPE"} \
    ${ENABLE_MULTIMODAL:+--enable-multimodal} \
    --mamba-scheduler-strategy "$MAMBA_SCHEDULER_STRATEGY" \
    ${MAMBA_SSM_DTYPE:+--mamba-ssm-dtype "$MAMBA_SSM_DTYPE"} \
    ${MAMBA_FULL_MEMORY_RATIO:+--mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO"} \
    ${MAX_MAMBA_CACHE_SIZE:+--max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"} \
    --schedule-policy "$SCHEDULE_POLICY" \
    ${SAMPLING_DEFAULTS:+--sampling-defaults "$SAMPLING_DEFAULTS"} \
    --num-continuous-decode-steps "$NUM_CONTINUOUS_DECODE_STEPS" \
    ${ENABLE_MIXED_CHUNK:+--enable-mixed-chunk} \
    ${ENABLE_PIECEWISE_CUDA_GRAPH:+--enable-piecewise-cuda-graph} \
    --log-level "$LOG_LEVEL" \
    ${LOG_REQUESTS:+--log-requests} \
    ${ENABLE_METRICS:+--enable-metrics} \
    "${SPEC_ARGS[@]}" \
    "$@"
