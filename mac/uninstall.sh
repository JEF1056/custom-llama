#!/usr/bin/env bash
#
# Removes the Qwen3.6-35B-A3B MLX LaunchAgent. Leaves downloaded models in ~/.qwen.
set -euo pipefail

LABEL=com.custom-llama.qwen36-mlx
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"

echo "[qwen36-mlx] LaunchAgent removed."
echo "[qwen36-mlx] Model/data left in ~/.qwen (delete manually to reclaim disk)."
