#!/usr/bin/env bash
#
# Removes the Bonsai MLX LaunchAgent. Leaves downloaded models in ~/.bonsai.
set -euo pipefail

LABEL=com.custom-llama.bonsai-mlx
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"

echo "[bonsai] LaunchAgent removed."
echo "[bonsai] Model/data left in ~/.bonsai (delete manually to reclaim disk)."
