#!/usr/bin/env bash
#
# Uninstalls MLX server LaunchAgents and cleans up processes/services.
#
# Usage:
#   bash mac/uninstall.sh [--purge-all] [--purge-cache]
#
set -euo pipefail

PURGE_ALL=0
PURGE_CACHE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --purge-all)   PURGE_ALL=1; shift ;;
        --purge-cache) PURGE_CACHE=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "[uninstall] Stopping and unloading LaunchAgents..."
LABELS=(
    "com.jfan.mlx-server"
    "com.custom-llama.qwen36-mlx"
    "com.custom-llama.qwen38-mlx"
    "com.custom-llama.qwen36-mlx.ml-2"
    "com.custom-llama.qwen36-mlx.ml-3"
)

UID_NUM=$(id -u)

for label in "${LABELS[@]}"; do
    launchctl bootout "gui/$UID_NUM/$label" 2>/dev/null || true
    launchctl disable "gui/$UID_NUM/$label" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/$label.plist"
done

# Clean any residual plists
find "$HOME/Library/LaunchAgents" -maxdepth 1 \( -name "*mlx-server*.plist" -o -name "*custom-llama*.plist" \) -delete 2>/dev/null || true

echo "[uninstall] Terminating any orphaned mlx server processes..."
pkill -f "mlx_vlm.server" 2>/dev/null || true
pkill -f "mlx_lm.server" 2>/dev/null || true

if [[ "$PURGE_CACHE" -eq 1 || "$PURGE_ALL" -eq 1 ]]; then
    echo "[uninstall] Cleaning MLX cache..."
    rm -rf "$HOME/.cache/mlx-vlm/caching"
fi

if [[ "$PURGE_ALL" -eq 1 ]]; then
    echo "[uninstall] Purging ~/.qwen..."
    rm -rf "$HOME/.qwen"
fi

echo "[uninstall] Complete."
