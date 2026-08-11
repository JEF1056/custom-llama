#!/usr/bin/env bash
#
# Benchmark script for MLX VLM backends (ml-2, ml-3) and ik_llama baseline.
# Sends requests with specific token counts and reports prefill/decode throughput.
#
# Usage:
#   bash mac/bench-mlx-vlm.sh [options]
#
# Options:
#   --tokens N       Tokens per request (default: 21000)
#   --server URL     Server URL (default: http://ml-2:8081)
#   --api-key KEY    API key (default: sk-noauth)
#   --help           Show this help

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
TOKENS_PER_REQUEST=21000
SERVER_URL="http://ml-2:8081"
API_KEY="sk-noauth"
MODEL="/Users/jfan/.qwen/models/qwen36-mlx"
PROMPT_FILE="/tmp/bench_prompt_vlm.txt"
REQUEST_FILE="/tmp/bench_request_vlm.json"
RESULT_FILE="/tmp/bench_result_vlm.json"

# ---- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens)   TOKENS_PER_REQUEST="$2";   shift 2 ;;
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
log() { printf '\033[1;36m[bench]\033[0m %s\n' "$*"; }

# ---- Generate a long prompt to a file ----------------------------------------
generate_prompt() {
    local tokens=$1
    python3 -c "
import random
random.seed(42)

paras = [
    'In the study of complex systems, researchers observe that emergent behavior arises from the interaction of many simple components, each following local rules without any central coordination. Consider a network of agents indexed by position, exchanging signals across noisy channels, adapting their internal state, and gradually converging toward a shared equilibrium that no single agent could have predicted in advance. This phenomenon appears across scales, from cellular automata to financial markets, suggesting universal principles of self-organization.',
    'The concept of attention mechanisms in deep learning has revolutionized natural language processing. Originally introduced in sequence-to-sequence models for machine translation, attention allows the model to focus on relevant parts of the input when producing each output token. Transformer architectures, which rely entirely on self-attention, have since become the dominant paradigm. Modern large language models use attention across billions of parameters, enabling remarkable capabilities in reasoning, translation, and code generation.',
    'Quantum computing promises to solve certain problems exponentially faster than classical computers. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in superposition states, representing both values simultaneously. When qubits are entangled, measuring one instantly affects the others regardless of distance. Quantum algorithms like Shor\'s factorization algorithm and Grover\'s search algorithm demonstrate this advantage, though building fault-tolerant quantum computers remains an enormous engineering challenge.',
    'Climate change research has advanced significantly through the use of Earth observation satellites and ground-based monitoring networks. Modern climate models incorporate data from thousands of sensors worldwide, tracking variables from atmospheric CO2 concentrations to ocean temperature gradients. The Intergovernmental Panel on Climate Change synthesizes findings from thousands of studies to project future climate scenarios under different emissions pathways. Recent observations show accelerating ice sheet loss and ocean acidification beyond earlier predictions.',
    'The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. Neural activity follows patterns that can be studied using functional MRI, electroencephalography, and increasingly, optogenetic techniques. Memory formation involves strengthening specific synaptic connections through mechanisms like long-term potentiation. The connectome project aims to map all neural connections in the brain, creating a comprehensive wiring diagram that could explain cognition, consciousness, and neurological disorders.',
    'Software engineering practices have evolved dramatically over the past five decades. From the early structured programming revolution to modern agile methodologies, each era brought new tools and philosophies. Version control systems like Git enable collaborative development across thousands of contributors. Continuous integration and deployment pipelines automate testing and release processes. Modern practices include test-driven development, code reviews, infrastructure as code, and microservice architectures that decompose monolithic applications into independently deployable services.',
    'Artificial intelligence ethics has become a critical field as AI systems gain influence in hiring, healthcare, criminal justice, and content moderation. Key concerns include bias in training data producing discriminatory outcomes, the black-box problem where models cannot explain their decisions, and the concentration of AI power in a few technology companies. Researchers and policymakers are developing frameworks for algorithmic accountability, transparency requirements, and fairness metrics. The pace of AI advancement often outstrips regulatory adaptation, creating governance challenges.',
    'Space exploration has entered a new era with reusable rockets, private space companies, and ambitious missions to the Moon and Mars. NASA\'s Artemis program aims to return humans to the lunar surface, establishing a permanent presence as a stepping stone for Mars missions. Private companies like SpaceX have dramatically reduced launch costs through reusable booster technology. Meanwhile, robotic missions continue to reveal the secrets of our solar system, from Perseverance rover findings on Mars to the detailed imaging of Jupiter\'s moon Europa by the Europa Clipper mission.',
    'Economic inequality has grown in many developed nations over the past four decades, with the top percentile capturing a disproportionate share of income growth. This trend coincides with declining union membership, globalization of manufacturing, and technological displacement of middle-skill jobs. Economists debate whether these patterns reflect skill-biased technological change, declining bargaining power of workers, or policy choices around taxation and social programs. Solutions proposed range from universal basic income to wealth taxes to retraining programs for displaced workers.',
    'The field of genomics has been transformed by next-generation sequencing technologies that can read an entire human genome in a single day for under a thousand dollars. This capability has enabled precision medicine approaches where treatments are tailored to individual genetic profiles. CRISPR gene editing allows precise modification of DNA sequences, with clinical trials already underway for sickle cell disease and certain cancers. The Human Genome Project\'s completion in 2003 opened the door to understanding the genetic basis of thousands of diseases.',
]

target_chars = int($tokens * 4.0)
chunks = []
total = 0
idx = 0
while total < target_chars:
    p = paras[idx % len(paras)]
    chunks.append(p)
    total += len(p)
    idx += 1

prompt = ' '.join(chunks)
# Ensure we hit approximately the target token count
actual_chars = len(prompt)
if actual_chars < target_chars * 0.9:
    extra = paras[0] * max(1, int((target_chars - actual_chars) / len(paras[0]) + 1))
    prompt += ' ' + extra

with open('$PROMPT_FILE', 'w') as f:
    f.write(prompt)

estimated_tokens = len(prompt) // 4
print(f'{estimated_tokens} estimated tokens, {len(prompt)} characters written to $PROMPT_FILE')
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
        {'role': 'assistant', 'content': 'Continue the analysis.'}
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

# ---- Run a single request and extract metrics --------------------------------
send_request() {
    local request_num=$1

    local start_time end_time elapsed http_code
    start_time=$(date +%s%N)

    http_code=$(curl -s -o "$RESULT_FILE" -w "%{http_code}" \
        --connect-timeout 300 \
        --max-time 600 \
        -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d @"$REQUEST_FILE" 2>/dev/null)

    end_time=$(date +%s%N)
    elapsed=$(( (end_time - start_time) / 1000000 ))  # milliseconds

    if [[ "$http_code" == "200" ]]; then
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
usage = data.get('usage', {})
timings = data.get('timings', {})
print(f'RESULT|http=$http_code|wall_ms=$elapsed|prompt_tokens={usage.get(\"prompt_tokens\", 0)}|completion_tokens={usage.get(\"completion_tokens\", 0)}|prompt_per_second={timings.get(\"prompt_per_second\", 0):.2f}|predicted_per_second={timings.get(\"predicted_per_second\", 0):.2f}|prompt_ms={timings.get(\"prompt_ms\", 0):.1f}|predicted_ms={timings.get(\"predicted_ms\", 0):.1f}')
"
    else
        echo "RESULT|http=$http_code|wall_ms=$elapsed|prompt_tokens=0|completion_tokens=0|prompt_per_second=0|predicted_per_second=0|prompt_ms=0|predicted_ms=0|ERROR"
    fi
}

# ---- Main benchmark ---------------------------------------------------------
log "========================================================"
log "MLX VLM Benchmark"
log "========================================================"
log "Server:      $SERVER_URL"
log "Model:       $MODEL"
log "Tokens/req:  $TOKENS_PER_REQUEST"
log "========================================================"

# Verify server is alive before starting
if ! check_server_alive; then
    log "ERROR: Server is not reachable at $SERVER_URL"
    exit 1
fi
log "Server is healthy before starting benchmarks."

# Generate prompt
log "Generating prompt (~${TOKENS_PER_REQUEST} tokens)..."
generate_prompt "$TOKENS_PER_REQUEST"
log ""

# Build JSON request
build_request
log ""

# Run single request
log "Sending request..."
RESULT=$(send_request 1)
log "Request complete."
log ""

# Parse and display results
log "========================================================"
log "RESULTS"
log "========================================================"
echo "$RESULT" | while IFS='|' read -r tag http wall_ms ptokens ctokens pps dpps pms dms extra; do
    echo "  HTTP Status:      $http"
    echo "  Wall Time:        ${wall_ms}ms"
    echo "  Prompt Tokens:    $ptokens"
    echo "  Completion Tokens: $ctokens"
    echo "  Prefill Throughput: ${pps} tok/s"
    echo "  Decode Throughput: ${dpps} tok/s"
    echo "  Prefill Time:     ${pms}ms"
    echo "  Decode Time:      ${dms}ms"
    if [[ "$extra" == "ERROR" ]]; then
        echo "  Status:           ERROR"
    else
        echo "  Status:           SUCCESS"
    fi
done
log "========================================================"
