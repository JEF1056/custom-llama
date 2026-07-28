#!/usr/bin/env bash
#
# MLX VLM Server Supervisor — ensures the server stays running 24/7.
# Monitors health via HTTP check; restarts on failure.
# All values are hardcoded — no env var dependencies at boot.
set -euo pipefail

MODEL_PATH="$HOME/.qwen/models/qwen36-mlx"
VENV_DIR="$HOME/.qwen/mlx-venv"
MLX_PORT="8080"
MLX_HOST="0.0.0.0"
LOG_DIR="$HOME/Library/Logs"
SUPERVISOR_LOG="$LOG_DIR/qwen36-mlx-supervisor.log"
HEALTH_CHECK_INTERVAL=30  # seconds
MAX_RESTARTS=10
RESTART_WINDOW=300  # 5 minutes

log() { printf '[%s] [qwen36-supervisor] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$SUPERVISOR_LOG"; }

# ---- Check model exists -----------------------------------------------------
if [[ ! -d "$MODEL_PATH" ]]; then
    log "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

# ---- Health check function --------------------------------------------------
check_health() {
    local retries=3
    local delay=2
    for ((i=1; i<=retries; i++)); do
        if curl -sf "http://localhost:${MLX_PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$delay"
    done
    return 1
}

# ---- Start server function --------------------------------------------------
start_server() {
    log "Starting mlx_vlm.server (port=$MLX_PORT, model=$MODEL_PATH)"
    
    cd "$MODEL_PATH"
    nohup bash -c "
        source '$VENV_DIR/bin/activate'
        exec python3 -m mlx_vlm.server \
            --host '$MLX_HOST' \
            --port $MLX_PORT \
            --model '$MODEL_PATH' \
            --kv-bits 4 \
            --max-kv-size 229376 \
            --prefill-step-size 1024 \
            --enable-thinking --kv-quant-scheme int4
    " >> "$LOG_DIR/qwen36-mlx.out.log" 2>>"$LOG_DIR/qwen36-mlx.err.log" &
    
    SERVER_PID=$!
    log "Server started (PID=$SERVER_PID)"
}

# ---- Main supervisor loop ---------------------------------------------------
log "=== Supervisor starting ==="
log "Health check interval: ${HEALTH_CHECK_INTERVAL}s"

RESTART_COUNT=0
RESTART_TIMES=()

while true; do
    # Wait for server to be ready
    sleep 5
    
    if ! check_health; then
        log "Health check FAILED — restarting (attempt $((RESTART_COUNT + 1)))"
        
        # Kill any existing server process
        pkill -f "mlx_vlm.server" 2>/dev/null || true
        sleep 2
        
        start_server
        RESTART_COUNT=$((RESTART_COUNT + 1))
        RESTART_TIMES+=($(date +%s))
        
        # Check if we're in a crash loop
        if [[ ${#RESTART_TIMES[@]} -gt $MAX_RESTARTS ]]; then
            # Count restarts in the window
            NOW=$(date +%s)
            RECENT=0
            for t in "${RESTART_TIMES[@]}"; do
                if (( NOW - t < RESTART_WINDOW )); then
                    RECENT=$((RECENT + 1))
                fi
            done
            
            if (( RECENT > MAX_RESTARTS )); then
                log "CRASH LOOP DETECTED: $RECENT restarts in ${RESTART_WINDOW}s — waiting longer"
                sleep 60
                # Reset counter
                RESTART_TIMES=()
                RESTART_COUNT=0
            fi
        fi
    else
        log "Health check OK"
        # Reset restart count on successful health check
        if (( RESTART_COUNT > 0 )); then
            log "Server recovered after $RESTART_COUNT restart(s)"
            RESTART_COUNT=0
            RESTART_TIMES=()
        fi
    fi
    
    sleep "$HEALTH_CHECK_INTERVAL"
done
