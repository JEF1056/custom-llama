#!/usr/bin/env bash
#
# Wrapper for the MLX VLM server: launches run-mlx-server.sh in a new
# Terminal window so the user can see the server output at startup.
# Invoked by the LaunchAgent on login / auto-restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/run-mlx-server.sh"

# Build the command to run inside the Terminal window.
# We cd into the model directory (as run-mlx-server.sh expects) and launch
# the server, keeping the window open when the server exits.
CMD="cd \"${MODEL_PATH:-$HOME/.qwen/models/qwen36-mlx}\" && exec bash -c \"source $VENV_DIR/bin/activate && exec $SERVER_SCRIPT\""

open -a Terminal.app --args -c "$CMD"
