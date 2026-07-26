#!/usr/bin/env bash
#
# Wrapper for the MLX VLM server: launches supervisor-mlx-server.sh as a
# background service. Invoked by the LaunchAgent on login / auto-restart.
# All values are hardcoded — no env var dependencies at boot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERVISOR_SCRIPT="$SCRIPT_DIR/supervisor-mlx-server.sh"

# Ensure log directory exists
mkdir -p "$HOME/Library/Logs"

# Launch the supervisor (which manages the server process)
nohup bash "$SUPERVISOR_SCRIPT" \
    >> "$HOME/Library/Logs/qwen36-mlx.out.log" 2>>"$HOME/Library/Logs/qwen36-mlx.err.log" &

SUPERVISOR_PID=$!
echo "[qwen36] Supervisor started (PID: $SUPERVISOR_PID)"
echo "[qwen36] Server: http://localhost:8080/v1"
echo "[qwen36] Logs: ~/Library/Logs/qwen36-mlx.out.log  ~/Library/Logs/qwen36-mlx.err.log"
echo "[qwen36] Supervisor: ~/Library/Logs/qwen36-mlx-supervisor.log"
