#!/usr/bin/env bash
#
# bench-mlx-longctx.sh — Enhanced long-context benchmark with real token counting,
#                        throughput reporting, streaming, and context sweep.
#
# Sends requests to the MLX VLM server (default: http://ml-2:8080) with
# configurable token counts, concurrent-like serial requests, and parses
# the server's timings object from the JSON response for throughput metrics.
#
# Usage:
#   # Basic 10k token single request:
#     bash mac/bench-mlx-longctx.sh
#
#   # 20k tokens, 5 requests, 2 cycles:
#     bash mac/bench-mlx-longctx.sh --tokens 20000 --requests 5 --cycles 2
#
#   # Streaming mode (per-token timing):
#     bash mac/bench-mlx-longctx.sh --stream --tokens 5000 --requests 2
#
#   # Context sweep from 4K to 256K:
#     bash mac/bench-mlx-longctx.sh --sweep --sweep-min 4096 --sweep-max 262144 --sweep-factor 2
#
#   # Custom server and model:
#     bash mac/bench-mlx-longctx.sh --server http://ml-3:8080 --api-key sk-mykey
#
# Prerequisites:
#   - SSH access to the target host (for multi-host)
#   - python3 on the local machine (for token counting / JSON parsing)
#   - The MLX server must be running and reachable
#
# Verification:
#   # Run a quick single-request check (should complete in <30s on a healthy server):
#     bash mac/bench-mlx-longctx.sh --tokens 512 --requests 1 --cycles 1
#
#   # Verify throughput reporting appears in output:
#     bash mac/bench-mlx-longctx.sh --tokens 1024 --requests 1 --cycles 1 2>&1 \
#       | grep -i 'prompt_tok/s\|decode_tok/s\|TTFT'
#
#   # Verify streaming mode shows token-level timing:
#     bash mac/bench-mlx-longctx.sh --stream --tokens 256 --requests 1 --cycles 1 \
#       2>&1 | head -40
#

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
TOKENS_PER_REQUEST=10000          # Target tokens in the prompt
REQUESTS_PER_CYCLE=3             # Requests per cycle
CYCLES=1                         # Number of cycles (repetitions)
SERVER_URL="http://ml-2:8080"   # Default to raw server port 8080
API_KEY="sk-noauth"
MODEL="qwen3.6-35b"
PROMPT_FILE="/tmp/bench_prompt_longctx.txt"
REQUEST_FILE="/tmp/bench_request_longctx.json"
RESPONSE_FILE="/tmp/bench_response_longctx.json"
STREAM=true                    # Default: do not stream (non-streaming for timing object)

# Context sweep parameters
SWEEP=false
SWEEP_MIN=4096
SWEEP_MAX=262144
SWEEP_FACTOR=2
SWEEP_TOKENS=()                # Populated by sweep range

# ---- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens)     TOKENS_PER_REQUEST="$2";   shift 2 ;;
        --requests)   REQUESTS_PER_CYCLE="$2";   shift 2 ;;
        --cycles)     CYCLES="$2";               shift 2 ;;
        --server)     SERVER_URL="$2";           shift 2 ;;
        --api-key)    API_KEY="$2";              shift 2 ;;
        --model)      MODEL="$2";                shift 2 ;;
        --no-stream)  STREAM=false;              shift   ;;
        --stream)     STREAM=true;               shift   ;;
        --sweep)      SWEEP=true; shift          ;;
        --sweep-min)  SWEEP_MIN="$2";            shift 2 ;;
        --sweep-max)  SWEEP_MAX="$2";            shift 2 ;;
        --sweep-factor) SWEEP_FACTOR="$2";       shift 2 ;;
        --help)
            sed -n '4,25p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Build sweep token list --------------------------------------------------
build_sweep_tokens() {
    SWEEP_TOKENS=()
    local t=$SWEEP_MIN
    while [[ $t -le $SWEEP_MAX ]]; do
        SWEEP_TOKENS+=("$t")
        t=$(( t * SWEEP_FACTOR ))
    done
    echo "Sweep token list: ${SWEEP_TOKENS[*]}"
}

# ---- Color helpers -----------------------------------------------------------
green()  { printf '\033[1;32m%s\033[0m' "$*"; }
red()    { printf '\033[1;31m%s\033[0m' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m' "$*"; }
cyan()   { printf '\033[1;36m%s\033[0m' "$*"; }
dim()    { printf '\033[0;90m%s\033[0m' "$*"; }

# ---- Ensure temp dir exists --------------------------------------------------
mkdir -p /tmp

# ---- Helper: tokenize prompt to get real token count --------------------------
# Calls the server's /v1/responses/input_tokens endpoint to get the actual
# number of tokens for a given prompt text, so we can adjust length.
get_token_count() {
    local prompt_text="$1"
    local result
    # Use python3 to safely JSON-encode the prompt text, avoiding injection
    # from quotes/backslashes in the prompt.
    result=$(python3 -c "
import json, sys, subprocess
prompt = sys.stdin.read()
payload = json.dumps({'model': sys.argv[1], 'input': [{'role': 'user', 'content': prompt}]})
proc = subprocess.run(
    ['curl', '-s', '--connect-timeout', '10', '--max-time', '30',
     '-X', 'POST', sys.argv[2],
     '-H', 'Content-Type: application/json',
     '-H', 'Authorization: Bearer ' + sys.argv[3],
     '-d', payload],
    capture_output=True, text=True
)
print(proc.stdout, end='')
" "$MODEL" "${SERVER_URL}/v1/responses/input_tokens" "$API_KEY" <<< "$prompt_text") || return 1

    # Extract token count from response: {"input_tokens": N}
    python3 -c "
import json, sys
try:
    data = json.loads(sys.argv[1])
    if 'input_tokens' in data:
        print(data['input_tokens'])
    elif 'tokens' in data:
        print(len(data['tokens']))
    elif 'count' in data:
        print(data['count'])
    elif 'n' in data:
        print(data['n'])
    elif 'num_tokens' in data:
        print(data['num_tokens'])
    else:
        # Fallback: try to find any numeric token count
        for k, v in data.items():
            if isinstance(v, list):
                print(len(v)); break
            elif isinstance(v, int):
                print(v); break
        else:
            print('0')
except Exception:
    print('0')
" "$result"
}

# ---- Helper: build prompt with real token count ------------------------------
# Generates a prompt and then refines it to hit the target token count
# by calling the server's tokenize endpoint.
generate_prompt_with_token_count() {
    local target_tokens=$1
    local prompt_text
    local actual_tokens=0
    local loop_count=0
    local max_loops=5

    # Generate an initial prompt (rough estimate)
    prompt_text=$(python3 -c "
words = 'The quick brown fox jumps over the lazy dog. Machine learning models process text and learn patterns from data. Each token represents a piece of the linguistic puzzle. Attention mechanisms are crucial for understanding long-range dependencies in sequences. Long context windows enable the model to remember earlier parts of a conversation or document. Trained on large-scale corpora, transformer architectures have shown remarkable performance across NLP benchmarks and vision tasks. Self-attention computes weighted combinations of all positions in the input sequence. This allows parallel processing and gradient flow across the entire sequence. The model generates predictions by autoregressively sampling the next token. Tokenization splits text into subword units using BPE or similar schemes. LoRA adapters modify pre-trained weights with minimal additional parameters. Quantization reduces memory footprint while maintaining accuracy. KV cache stores attention keys and values for efficient decoding. Speculative decoding uses a smaller draft model to propose tokens. Early exit mechanisms allow faster responses for confident predictions. Multimodal models accept both text and image inputs via cross-attention layers. Flash attention optimizes the attention mechanism memory access pattern. RoPE positional encoding provides relative position information. The tokenizer maps input strings to token IDs via a vocabulary lookup table. Embedding projections map token IDs to dense vector representations. Layer normalization stabilizes training by normalizing activations. Residual connections skip over transformations to preserve gradient flow. The final linear projection maps hidden representations to the vocabulary space. '
# Use a slightly aggressive ratio to account for short words being subword tokens
num_chunks = int($target_tokens / 1.2)
prompts = []
i = 0
while i < num_chunks and len(prompts) < num_chunks:
    prompts.append(words)
    i += 1
result = ''.join(prompts)
print(result[:200000], end='')
" 2>/dev/null)

    # Iterate: tokenize, check, adjust
    for ((loop_count=0; loop_count<max_loops; loop_count++)); do
        actual_tokens=$(get_token_count "$prompt_text" 2>/dev/null) || actual_tokens=0

        if [[ $actual_tokens -ge $(( target_tokens - 500 )) ]] && \
           [[ $actual_tokens -le $(( target_tokens + 500 )) ]]; then
            # Close enough — trim to exact target
            prompt_text=$(python3 -c "
text = '''$(echo "$prompt_text" | head -c 200000)'''
token_chars_per_token = len(text.strip()) / max(1, $actual_tokens)
target_len = int($target_tokens * token_chars_per_token)
# Find a sensible word boundary near the target length
trimmed = text[:target_len]
last_space = trimmed.rfind(' ')
if last_space > 0:
    trimmed = trimmed[:last_space]
print(trimmed, end='')
" 2>/dev/null)
            # Final recount
            actual_tokens=$(get_token_count "$prompt_text" 2>/dev/null) || actual_tokens=$actual_tokens
            break
        elif [[ $actual_tokens -lt $(( target_tokens - 500 )) ]]; then
            # Add more content
            extra=$(python3 -c "
extra_words = ' ' * 500
import random; random.seed(42)
result = ''
it = 0
base = 'Artificial intelligence continues to transform how we process and generate language. Research in neural networks has advanced significantly with improved training methodologies and larger datasets. Computational efficiency remains a key focus for deployment at scale. Model compression techniques like pruning and quantization enable running large models on edge devices. Multi-task learning allows a single model to perform well across multiple domains. The training corpus for modern language models includes billions of tokens from diverse sources. Transformer architectures allow parallel computation during training but require significant memory. The attention mechanism computes pairwise interactions between all token positions. Positional encodings allow the model to understand relative ordering. Normalization layers stabilize the training process by reducing internal covariate shift. '
while it < 100:
    result += base
    it += 1
print(result, end='')
")
            prompt_text="${prompt_text}${extra}"
        else
            # Slightly over — trim
            prompt_text=$(python3 -c "
text = '''$(echo "$prompt_text" | head -c 200000)'''
token_chars_per_token = len(text.strip()) / max(1, $actual_tokens)
target_len = int($target_tokens * token_chars_per_token)
trimmed = text[:target_len]
last_space = trimmed.rfind(' ')
if last_space > 0:
    trimmed = trimmed[:last_space]
print(trimmed, end='')
" 2>/dev/null)
        fi
    done

    # Final recount
    actual_tokens=$(get_token_count "$prompt_text" 2>/dev/null) || actual_tokens=${TOKENS_PER_REQUEST}
    # Write to file
    echo -n "$prompt_text" > "$PROMPT_FILE"
    echo "$actual_tokens" > /tmp/bench_actual_tokens.txt
    echo "$actual_tokens"
}

# ---- Helper: build JSON request ----------------------------------------------
build_request() {
    local prompt_file="$PROMPT_FILE"
    local num_tokens
    num_tokens=$(cat /tmp/bench_actual_tokens.txt 2>/dev/null || echo "$TOKENS_PER_REQUEST")

    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    prompt = f.read()
stream_val = sys.argv[2].lower() == 'true'
request = {
    'model': sys.argv[3],
    'messages': [
        {'role': 'user', 'content': prompt}
    ],
    'max_tokens': 1024,
    'temperature': 0.6,
    'top_p': 0.95,
    'stream': stream_val
}
with open(sys.argv[4], 'w') as f:
    json.dump(request, f)
print(f'Request built — {sys.argv[5]} tokens in prompt, stream={stream_val}')
" "$prompt_file" "$STREAM" "$MODEL" "$REQUEST_FILE" "$num_tokens" 2>/dev/null
}

# ---- Helper: parse timings from response -------------------------------------
# Parses the mlx-vlm server response to extract throughput metrics.
# The server returns a 'timings' object with:
#   prompt_per_second, predicted_per_second, prompt_ms, predicted_ms,
#   prompt_tokens, completion_tokens
parse_timings() {
    local response_file="$RESPONSE_FILE"

    if [[ ! -f "$response_file" ]]; then
        echo "0|0|0|0|0|0|0"
        return
    fi

    python3 -c "
import json, sys
try:
    with open('$response_file') as f:
        data = json.loads(f.read())
except Exception:
    print('0|0|0|0|0|0|0')
    sys.exit(0)

# Navigate to timings object (may be nested under choices[0].usage)
timings = {}
if isinstance(data, dict):
    # Direct timings key
    if 'timings' in data:
        t = data['timings']
        timings = t if isinstance(t, dict) else {}
    # mlx-vlm server wraps in a response with choices
    elif 'choices' in data:
        first = data['choices'][0] if data['choices'] else {}
        if 'timings' in first:
            t = first['timings']
            timings = t if isinstance(t, dict) else {}
        elif 'delta' and 'timings' in first.get('delta', {}):
            t = first['delta'].get('timings', {})
            timings = t if isinstance(t, dict) else {}
    # Streaming: each chunk may have timings
    elif 'content' in data and 'timings' in data:
        t = data['timings']
        timings = t if isinstance(t, dict) else {}

# Also look for usage in completion
usage = {}
if isinstance(data, dict):
    if 'usage' in data:
        usage = data.get('usage', {})
    elif 'choices' in data:
        first = data['choices'][0] if data['choices'] else {}
        if 'usage' in first:
            usage = first['usage']

prompt_tok = (timings.get('prompt_tokens') or 0)
completion_tok = (timings.get('completion_tokens') or 0)
prompt_ts = timings.get('prompt_per_second') or 0
decode_ts = timings.get('predicted_per_second') or 0
prompt_ms = timings.get('prompt_ms') or 0
decode_ms = timings.get('predicted_ms') or 0
sample_time = 0  # TTFT estimate (first prompt_ms or rapidly early token gap)

print(f'{prompt_tok}|{completion_tok}|{prompt_ts:.2f}|{decode_ts:.2f}|{prompt_ms:.1f}|{decode_ms:.1f}|{sample_time:.1f}')
" 2>/dev/null || echo "0|0|0.00|0.00|0.0|0.0|0.0"
}

# ---- Helper: process streaming response --------------------------------------
# For streaming mode, measures time between each token emitted.
stream_response() {
    local start_time=$(date +%s%N)
    local line_num=0

    curl -s --connect-timeout 300 --max-time 600 -N \
        -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d @"$REQUEST_FILE" 2>/dev/null | \
    while IFS= read -r line; do
        line_num=$((line_num + 1))
        # Parse SSE data lines for token timing
        if echo "$line" | grep -q "^data:"; then
            local data_line=$(echo "$line" | sed 's/^data: //')
            local token_content
            token_content=$(python3 -c "
import json, sys
try:
    chunk = json.loads(sys.argv[1])
    choices = chunk.get('choices', [])
    if choices:
        delta = choices[0].get('delta', {})
        content = delta.get('content', '')
        if content:
            print(content)
    timings = chunk.get('timings', {}) or {}
    if chunk.get('timings'):
        # We also print timing info for the last chunk
        p = timings.get('prompt_per_second', 0)
        d = timings.get('predicted_per_second', 0)
        t = chunk.get('timings', {})
        print(f'[timings] pps={p:.2f}, dps={d:.2f}', file=sys.stderr)
except:
    pass
" "$data_line" 2>/dev/null)
            if [[ -n "$token_content" ]]; then
                local cur_time=$(date +%s%N)
                local elapsed_ms=$(( (cur_time - start_time) / 1000000 ))
                dim "  [token ${line_num}] ${token_content:0:80}"
                start_time=$cur_time
            fi
        fi
    done
}

# ---- Helper: send a single request with timing --------------------------------
send_request() {
    local request_num=$1
    local cycle_num=$2

    local start_time end_time elapsed http_code
    start_time=$(date +%s%N)

    if [[ "$STREAM" == true ]]; then
        # Streaming mode — time token-by-token
        echo "  $start_time 0" > /tmp/llm_tokens.csv
        stream_response
        local end_stream=$(date +%s%N)
        elapsed=$(( (end_stream - start_time) / 1000000 ))
        total_elapsed=$((total_elapsed + elapsed))
        total_requests=$((total_requests + 1))

        echo "  -> Total streaming time: ${elapsed}ms"
    else
        # Non-streaming: get full response with timings object
        http_code=$(curl -s -o "$RESPONSE_FILE" -w "%{http_code}" \
            --connect-timeout 300 \
            --max-time 600 \
            -X POST "${SERVER_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${API_KEY}" \
            -d @"$REQUEST_FILE" 2>/dev/null)

        end_time=$(date +%s%N)
        elapsed=$(( (end_time - start_time) / 1000000 ))

        if [[ "$http_code" == "200" ]]; then
            local timings_parsed
            timings_parsed=$(parse_timings)
            local prompt_tok completion_tok prompt_ts decode_ts prompt_ms decode_ms sample_ms
            prompt_tok=$(echo "$timings_parsed" | cut -d'|' -f1)
            completion_tok=$(echo "$timings_parsed" | cut -d'|' -f2)
            prompt_ts=$(echo "$timings_parsed" | cut -d'|' -f3)
            decode_ts=$(echo "$timings_parsed" | cut -d'|' -f4)
            prompt_ms=$(echo "$timings_parsed" | cut -d'|' -f5)
            decode_ms=$(echo "$timings_parsed" | cut -d'|' -f6)
            sample_ms="${timings_parsed##*|}"

            echo "$prompt_tok|${completion_tok}|${prompt_ts}|${decode_ts}|${prompt_ms}|${decode_ms}|${elapsed}" \
                >> /tmp/bench_per_request.csv

            total_elapsed=$((total_elapsed + elapsed))
            total_requests=$((total_requests + 1))
            total_responses="${total_responses}${timings_parsed}\n"

            yellow "  #${request_num} [cycle ${cycle_num}]: OK (${elapsed}ms)"
            dim "       TTFT(prepurchase_err): ~${prompt_ms}ms"
            dim "       prompt_tok/s:  ${prompt_ts}, decode_tok/s:  ${decode_ts}"
            dim "       prompt tokns:  ${prompt_tok}, completion toks: ${completion_tok}"
        else
            local error_msg="HTTP $http_code (${elapsed}ms)"
            red "  #${request_num} [cycle ${cycle_num}]: FAIL — $error_msg"
            failed=$((failed + 1))
            total_requests=$((total_requests + 1))
        fi
    fi
}

# ---- Check if server is alive ------------------------------------------------
check_server_alive() {
    curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 10 "${SERVER_URL}/health" 2>/dev/null | grep -q "200"
}

# ---- Aggregate report --------------------------------------------------------
accumulate_and_report() {
    local csv_file="/tmp/bench_per_request.csv"
    python3 -c "
import sys

results = []
try:
    with open('$csv_file') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 7:
                continue
            prompt_tok = int(float(parts[0])) if parts[0] and parts[0] != 'None' else 0
            completion_tok = int(float(parts[1])) if parts[1] and parts[1] != 'None' else 0
            prompt_ts = float(parts[2]) if parts[2] and parts[2] != 'None' and float(parts[2]) > 0 else None
            decode_ts = float(parts[3]) if parts[3] and parts[3] != 'None' and float(parts[3]) > 0 else None
            prompt_ms = float(parts[4]) if parts[4] and parts[4] != 'None' else 0
            decode_ms = float(parts[5]) if parts[5] and parts[5] != 'None' else 0
            total_ms = float(parts[6]) if parts[6] and parts[6] != 'None' else 0

            results.append({
                'prompt_tok': prompt_tok,
                'completion_tok': completion_tok,
                'prompt_ts': prompt_ts,
                'decode_ts': decode_ts,
                'prompt_ms': prompt_ms,
                'decode_ms': decode_ms,
                'total_ms': total_ms,
            })
except FileNotFoundError:
    pass

if not results:
    print('No data collected.')
    sys.exit(0)

print()
print('=' * 72)
if '$SWEEP':
    print('CONTEXT SWEEP RESULTS')
    print('=' * 72)
else:
    print('PER-REQUEST BREAKDOWN')
    print('=' * 72)
print(f\"{'#':>3s}  {'PromptTok':>10s}  {'CompTok':>10s}  {'PromptTok/s':>12s}  {'DecodeTok/s':>13s}  {'TTFT':>10s}  {'DecodeMs':>10s}  {'TotalMs':>10s}\")
print('-' * 72)

for i, r in enumerate(results, 1):
    pt_s = f'{r[\"prompt_ts\"]:.1f}' if r['prompt_ts'] else 'N/A'
    dk_s = f'{r[\"decode_ts\"]:.1f}' if r['decode_ts'] else 'N/A'
    ttft_s = f'{r[\"prompt_ms\"]:.0f}ms'
    dkms_s = f'{r[\"decode_ms\"]:.0f}ms'
    total_s = f'{r[\"total_ms\"]:.0f}ms'
    print(f'{i:>3d}  {r[\"prompt_tok\"]:>10d}  {r[\"completion_tok\"]:>10d}  {pt_s:>12s}  {dk_s:>13s}  {ttft_s:>10s}  {dkms_s:>10s}  {total_s:>10s}')

print('=' * 72)

# Aggregate stats
total_prompt_tok = sum(r['prompt_tok'] for r in results)
total_completion_tok = sum(r['completion_tok'] for r in results)
total_time_ms = sum(r['total_ms'] for r in results)

# Compute aggregate throughput (decode tokens per second)
total_completion_seconds = total_time_ms / 1000.0 if total_time_ms > 0 else 0
avg_decode_tok_s = total_completion_tok / total_completion_seconds if total_completion_seconds > 0 else 0

# Compute avg per-request throughput (when available)
valid_decode = [r['decode_ts'] for r in results if r['decode_ts']]
avg_decode_ts = sum(valid_decode) / len(valid_decode) if valid_decode else 0
valid_prompt = [r['prompt_ts'] for r in results if r['prompt_ts']]
avg_prompt_ts = sum(valid_prompt) / len(valid_prompt) if valid_prompt else 0
avg_ttft = sum(r['prompt_ms'] for r in results) / len(results) if results else 0

print()
print('AGGREGATE STATISTICS')
print('-' * 72)
print(f'  Total requests:                {len(results)}')
print(f'  Total prompt tokens (input):   {total_prompt_tok:>10d}')
print(f'  Total completion tokens:       {total_completion_tok:>10d}')
print(f'  Total wall-clock time:         {total_time_ms:>10d}ms ({total_completion_seconds:.1f}s)')
print(f'  Aggregate decode throughput:   {avg_decode_tok_s:>10.1f} tok/s')
print(f'  Average prompt throughput:     {avg_prompt_ts:>10.1f} tok/s')
print(f'  Average decode throughput:     {avg_decode_ts:>10.1f} tok/s')
print(f'  Average TTFT (prefill):        {avg_ttft:>10.0f}ms')
print(f'  Average total latency:         {total_time_ms / len(results):>10.0f}ms')
print('=' * 72)
" 2>/dev/null

    # Cleanup
    rm -f "$csv_file"
}

# ---- Main -------------------------------------------------------------------
echo ""
cyan "================================================================"
cyan "  MLX Long-Context Benchmark"
cyan "================================================================"

# Make temp files
touch /tmp/bench_per_request.csv
: > /tmp/bench_per_request.csv   # clear

if [[ "$SWEEP" == true ]]; then
    build_sweep_tokens
fi

# ---- Slow per-request stats collection starting
total_elapsed=0
total_requests=0
failed=0
total_responses=""

# ---- Run the benchmark -------------------------------------------------------
if [[ "$SWEEP" == true ]]; then
    # ---- Context sweep mode --------------------------------------------------
    echo "  Sweep mode: scanning contexts from ${SWEEP_MIN}K to ${SWEEP_MAX}K"
    echo "  Sweep factor: x${SWEEP_FACTOR}"
    echo "  Tokens: requested ${TOKENS_PER_REQUEST}, will override per sweep level"
    echo ""

    for sweep_tokens in "${SWEEP_TOKENS[@]}"; do
        echo "--------------------------------------------------------"
        echo "  SWEEP LEVEL: ${sweep_tokens} tokens"
        echo "--------------------------------------------------------"
        TOKENS_PER_REQUEST=$sweep_tokens

        # Generate prompt with real token count
        echo "  Generating prompt (~${sweep_tokens} target tokens)..."
        generate_prompt_with_token_count "$sweep_tokens" || true
        local_actual=$(cat /tmp/bench_actual_tokens.txt 2>/dev/null || echo "$sweep_tokens")
        echo "  Actual token count: ${local_actual}"

        # Build request
        build_request

        # Run requests for this sweep level
        for cycle in $(seq 1 $CYCLES); do
            for req in $(seq 1 $REQUESTS_PER_CYCLE); do
                send_request "$req" "$cycle"
            done
        done

        echo "  Sweep level ${sweep_tokens}K completed."
        echo ""
    done
else
    # ---- Standard benchmark mode -----------------------------------------------
    echo "  Server:        $SERVER_URL"
    echo "  Model:         $MODEL"
    echo "  Tokens/req:    ${TOKENS_PER_REQUEST}"
    echo "  Requests/cycle: ${REQUESTS_PER_CYCLE}"
    echo "  Cycles:        ${CYCLES}"
    echo "  Streaming:     $STREAM"
    echo "  API key:       ${API_KEY:0:8}..."
    echo ""

    # Generate prompt with real token count
    echo "  Generating prompt (~${TOKENS_PER_REQUEST} target tokens)..."
    local_actual=$(generate_prompt_with_token_count "$TOKENS_PER_REQUEST")
    echo "  Actual token count: ${local_actual}"
    echo ""

    build_request
    echo "  Request payload built."
    echo ""

    # Run benchmark cycles
    for cycle in $(seq 1 $CYCLES); do
        echo "--------------------------------------------------------"
        echo "  CYCLE $cycle of $CYCLES"
        echo "--------------------------------------------------------"

        for req in $(seq 1 $REQUESTS_PER_CYCLE); do
            send_request "$req" "$cycle"

            # Small delay between requests to avoid overwhelming
            sleep 1
        done
        echo ""

        # Check server health after each cycle
        if ! check_server_alive; then
            red "  WARNING: Server became unhealthy after cycle $cycle."
            break
        fi
    done
fi

# ---- Aggregate report --------------------------------------------------------
echo ""
accumulate_and_report

# ---- Final summary -----------------------------------------------------------
echo ""
green "================================================================"
green "  BENCHMARK SUMMARY"
green "================================================================"
echo "  Total requests sent:     $total_requests"
echo "  Failed requests:         $failed"
if [[ $total_requests -gt 0 ]]; then
    echo "  Success rate:            $(( (total_requests - failed) * 100 / total_requests ))%"
    echo "  Total wall-clock time:   ${total_elapsed}ms"
    echo "  Average per-request:     $(( total_elapsed / total_requests ))ms"
fi
green "================================================================"

# Cleanup temp files
rm -f "$PROMPT_FILE" "$REQUEST_FILE" "$RESPONSE_FILE" "/tmp/bench_actual_tokens.txt"

if [[ $failed -gt 0 ]]; then
    red "  BENCHMARK FAILED: $failed request(s) failed"
    exit 1
else
    green "  BENCHMARK PASSED: all $total_requests requests succeeded"
    exit 0
fi
