#!/usr/bin/env bash
#
# Wrapper for the MLX VLM server: launches run-mlx-server.sh as a background
# service. Invoked by the LaunchAgent on login / auto-restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/run-mlx-server.sh"

# Run the server in the background, redirecting output to logs
cd "${MODEL_PATH:-$HOME/.qwen/models/qwen36-mlx}"
export MODEL_PATH="${MODEL_PATH:-$HOME/.qwen/models/qwen36-mlx}"
export VENV_DIR="${VENV_DIR:-$HOME/.qwen/mlx-venv}"
export MLX_PORT="${MLX_PORT:-8081}"
export MLX_MAX_KV_SIZE="${MLX_MAX_KV_SIZE:-229376}"
nohup bash -c "source \$VENV_DIR/bin/activate && exec \$SERVER_SCRIPT" \
    >> "$HOME/Library/Logs/qwen36-mlx.out.log" 2>&1 &

echo "[qwen36] Server started in background (PID: $!)"
echo "[qwen36] Logs: ~/Library/Logs/qwen36-mlx.out.log  ~/Library/Logs/qwen36-mlx.err.log"
