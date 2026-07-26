#!/usr/bin/env bash
#
# Wrapper for the MLX VLM server: launches supervisor-mlx-server.sh as a
# background service. Invoked by the LaunchAgent on login / auto-restart.
# The supervisor ensures the server stays running 24/7 with health checks
# and automatic restarts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERVISOR_SCRIPT="$SCRIPT_DIR/supervisor-mlx-server.sh"

export MODEL_PATH="${MODEL_PATH:-$HOME/.qwen/models/qwen36-mlx/quantized}"
export VENV_DIR="${VENV_DIR:-$HOME/.qwen/mlx-venv}"
export MLX_PORT="${MLX_PORT:-8081}"
export MLX_HOST="${MLX_HOST:-0.0.0.0}"
export MLX_MAX_KV_SIZE="${MLX_MAX_KV_SIZE:-229376}"
export MLX_KV_BITS="${MLX_KV_BITS:-3}"
export MLX_PREFILL_STEP_SIZE="${MLX_PREFILL_STEP_SIZE:-1024}"

# Ensure log directory exists
mkdir -p "$HOME/Library/Logs"

# Launch the supervisor (which manages the server process)
nohup bash "$SUPERVISOR_SCRIPT" \
    >> "$HOME/Library/Logs/qwen36-mlx.out.log" 2>>"$HOME/Library/Logs/qwen36-mlx.err.log" &

SUPERVISOR_PID=$!
echo "[qwen36] Supervisor started (PID: $SUPERVISOR_PID)"
echo "[qwen36] Server: http://localhost:${MLX_PORT}/v1"
echo "[qwen36] Logs: ~/Library/Logs/qwen36-mlx.out.log  ~/Library/Logs/qwen36-mlx.err.log"
echo "[qwen36] Supervisor: ~/Library/Logs/qwen36-mlx-supervisor.log"
