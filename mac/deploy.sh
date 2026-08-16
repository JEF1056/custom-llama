#!/usr/bin/env bash
#
# Deploy the MLX VLM server to Mac machines (ml-2, ml-3).
# All servers serve on port 8080.
#
# Usage:
#   bash mac/deploy.sh ml-2 ml-3
#   bash mac/deploy.sh --uninstall ml-2 ml-3
#
set -euo pipefail

ALL_HOSTS=(ml-2 ml-3)
ACTION="deploy"
HOSTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --uninstall|--clean) ACTION="uninstall"; shift ;;
        --deploy)            ACTION="deploy"; shift ;;
        *)                   HOSTS+=("$1"); shift ;;
    esac
done

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    HOSTS=("${ALL_HOSTS[@]}")
fi

green()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
red()    { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Pre-flight check
for host in "${HOSTS[@]}"; do
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" &>/dev/null; then
        red "ERROR: Cannot reach $host via SSH."
        exit 1
    fi
done

green "Pre-flight OK: all target hosts reachable."

if [[ "$ACTION" == "uninstall" ]]; then
    for host in "${HOSTS[@]}"; do
        yellow "Cleaning up and uninstalling on $host..."
        ssh "$host" "bash -s" < "$REPO_ROOT/mac/uninstall.sh"
        green "  $host uninstalled successfully."
    done
    exit 0
fi

# Deploy
FAILED=()
for host in "${HOSTS[@]}"; do
    LABEL="com.custom-llama.qwen38-mlx"
    yellow "Deploying to $host..."

    # Ensure remote dirs exist
    ssh "$host" "mkdir -p ~/.qwen/custom-llama/mac ~/.qwen/models ~/Library/LaunchAgents"

    # Sync mac scripts
    scp "$REPO_ROOT/mac/run-mlx-server.sh" "$REPO_ROOT/mac/uninstall.sh" "$REPO_ROOT/mac/com.custom-llama.qwen38-mlx.plist.template" "$host:~/.qwen/custom-llama/mac/"

    # Remote setup and launch
    ssh "$host" <<'REMOTE_EOF'
set -euo pipefail
UID_NUM=$(id -u)
LABEL="com.custom-llama.qwen38-mlx"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

# Stop legacy or previous services
launchctl bootout "gui/$UID_NUM/com.jfan.mlx-server" 2>/dev/null || true
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
pkill -f "mlx_vlm.server" 2>/dev/null || true

# Render plist
sed -e "s|__REPO__|$HOME/.qwen/custom-llama|g" \
    -e "s|__HOME__|$HOME|g" \
    -e "s|__LABEL__|$LABEL|g" \
    "$HOME/.qwen/custom-llama/mac/com.custom-llama.qwen38-mlx.plist.template" > "$PLIST_DST"

# Bootstrap and kickstart LaunchAgent
chmod +x "$HOME/.qwen/custom-llama/mac/run-mlx-server.sh"
chmod +x "$HOME/.qwen/custom-llama/mac/uninstall.sh"
launchctl bootstrap "gui/$UID_NUM" "$PLIST_DST"
launchctl enable "gui/$UID_NUM/$LABEL"
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true
REMOTE_EOF

    # Verify service
    yellow "  Waiting for $host to become healthy..."
    HEALTHY=0
    for i in {1..20}; do
        if curl -fsS --connect-timeout 2 "http://${host}:8080/v1/models" &>/dev/null || curl -fsS --connect-timeout 2 "http://${host}:8080/health" &>/dev/null; then
            HEALTHY=1
            break
        fi
        sleep 2
    done

    if [[ "$HEALTHY" -eq 1 ]]; then
        green "  $host: DEPLOYED & HEALTHY (http://${host}:8080/v1)"
    else
        red "  $host: Deployed, but waiting for model load. Check logs: ~/.qwen/mlx-server.log"
    fi
done
