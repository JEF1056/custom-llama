#!/usr/bin/env bash
#
# Stress-test the MLX server on ml-2 with long-context requests.
# Sends N requests of M tokens each, repeating K cycles. Reports failures,
# latency, and whether the server stayed alive throughout.
#
# Usage:
#   bash mac/bench-mlx-stress.sh [options]
#
# Options:
#   --tokens N       Tokens per request (default: 16384)
#   --requests N     Requests per cycle (default: 5)
#   --cycles N       Number of cycles (default: 3)
#   --server URL     Server URL (default: http://ml-2:8081)
#   --api-key KEY    API key (default: sk-noauth)
#   --model NAME     Model name (default: llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved)
#   --help           Show this help

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
TOKENS_PER_REQUEST=16384
REQUESTS_PER_CYCLE=5
CYCLES=3
SERVER_URL="http://ml-2:8081"
API_KEY="sk-noauth"
MODEL="/Users/jfan/.qwen/models/qwen36-mlx"
PROMPT_FILE="/tmp/bench_prompt.txt"
REQUEST_FILE="/tmp/bench_request.json"

# ---- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens)   TOKENS_PER_REQUEST="$2";   shift 2 ;;
        --requests) REQUESTS_PER_CYCLE="$2";   shift 2 ;;
        --cycles)   CYCLES="$2";               shift 2 ;;
        --server)   SERVER_URL="$2";           shift 2 ;;
        --api-key)  API_KEY="$2";              shift 2 ;;
        --model)    MODEL="$2";                shift 2 ;;
        --help)
            sed -n '4,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Helpers ----------------------------------------------------------------
PASS=0
FAIL=0
TOTAL_LATENCY=0
TOTAL_REQUESTS=0
CYCLE=0
SERVER_ALIVE=true

log() { printf '\033[1;36m[bench]\033[0m %s\n' "$*"; }
pass() { PASS=$((PASS + 1)); TOTAL_REQUESTS=$((TOTAL_REQUESTS + 1)); }
fail() { FAIL=$((FAIL + 1)); TOTAL_REQUESTS=$((TOTAL_REQUESTS + 1)); log "FAIL #$FAIL: $*"; SERVER_ALIVE=false; }

# ---- Generate a long prompt to a file ----------------------------------------
generate_prompt() {
    local tokens=$1
    python3 -c "
words = 'The quick brown fox jumps over the lazy dog. '
num_words = int($tokens / 1.3)
prompt = words * num_words
with open('$PROMPT_FILE', 'w') as f:
    f.write(prompt)
print(len(prompt), 'characters written to', '$PROMPT_FILE')
"
}

# ---- Check if server is alive ------------------------------------------------
check_server_alive() {
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${SERVER_URL}/health" 2>/dev/null)
    [[ "$status" == "200" ]]
}

# ---- Build JSON request with prompt from file --------------------------------
build_request() {
    python3 -c "
import json
with open('$PROMPT_FILE') as f:
    prompt = f.read()
request = {
    'model': '$MODEL',
    'messages': [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': 'Continue the story.'}
    ],
    'max_tokens': 512,
    'temperature': 0.6,
    'top_p': 0.95
}
with open('$REQUEST_FILE', 'w') as f:
    json.dump(request, f)
print('Request payload built:', len(prompt), 'chars')
"
}

# ---- Run a single request ---------------------------------------------------
send_request() {
    local request_num=$1

    local start_time end_time elapsed http_code

    start_time=$(date +%s%N)

    http_code=$(curl -s -o /tmp/bench_response.json -w "%{http_code}" \
        --connect-timeout 300 \
        --max-time 600 \
        -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d @"$REQUEST_FILE" 2>/dev/null)

    end_time=$(date +%s%N)
    elapsed=$(( (end_time - start_time) / 1000000 ))  # milliseconds
    TOTAL_LATENCY=$((TOTAL_LATENCY + elapsed))

    if [[ "$http_code" == "200" ]]; then
        pass
        log "Request #$request_num: OK (${elapsed}ms)"
    else
        local error_msg
        error_msg=$(python3 -c "
import json, sys
try:
    data = json.load(open('/tmp/bench_response.json'))
    print(json.dumps(data.get('detail', str(data)))[:200])
except:
    print('Unknown error')
" 2>/dev/null || echo "HTTP $http_code")
        fail "HTTP $http_code (${elapsed}ms): $error_msg"
    fi
}

# ---- Main benchmark loop ----------------------------------------------------
log "========================================================"
log "MLX Server Stress Test"
log "========================================================"
log "Server:      $SERVER_URL"
log "Model:       $MODEL"
log "Tokens/req:  $TOKENS_PER_REQUEST"
log "Requests/cycle: $REQUESTS_PER_CYCLE"
log "Cycles:      $CYCLES"
log "API key:     ${API_KEY:0:8}..."
log "========================================================"

# Verify server is alive before starting
if ! check_server_alive; then
    log "ERROR: Server is not reachable at $SERVER_URL"
    exit 1
fi
log "Server is healthy before starting benchmarks."
log ""

# Generate prompt to file
log "Generating long prompt (~${TOKENS_PER_REQUEST} tokens)..."
generate_prompt "$TOKENS_PER_REQUEST"
log ""

# Build JSON request
build_request
log "Request payload built."
log ""

# Run benchmark cycles
for cycle in $(seq 1 $CYCLES); do
    CYCLE=$((CYCLE + 1))
    log "--------------------------------------------------------"
    log "CYCLE $CYCLE of $CYCLES"
    log "--------------------------------------------------------"

    for req in $(seq 1 $REQUESTS_PER_CYCLE); do
        send_request "$req"

        # Check server health after each request
        if ! check_server_alive; then
            log "Server died after request #$req in cycle $cycle!"
            SERVER_ALIVE=false
            break 2
        fi

        # Small delay between requests to avoid overwhelming
        sleep 2
    done

    if [[ "$SERVER_ALIVE" == false ]]; then
        log "Server became unhealthy during cycle $cycle."
        break
    fi
done

# ---- Summary ----------------------------------------------------------------
log ""
log "========================================================"
log "RESULTS"
log "========================================================"
if [[ $TOTAL_REQUESTS -gt 0 ]]; then
    AVG_LATENCY=$((TOTAL_LATENCY / TOTAL_REQUESTS))
else
    AVG_LATENCY=0
fi

log "Total requests:    $TOTAL_REQUESTS"
log "Passed:            $PASS"
log "Failed:            $FAIL"
log "Avg latency:       ${AVG_LATENCY}ms"
log "Server stable:     $SERVER_ALIVE"
log "========================================================"

if [[ $FAIL -gt 0 ]]; then
    log "STATUS: $FAIL failure(s) detected"
    exit 1
elif [[ "$SERVER_ALIVE" == false ]]; then
    log "STATUS: Server became unstable"
    exit 1
else
    log "STATUS: All $PASS requests passed, server stable"
    exit 0
fi
