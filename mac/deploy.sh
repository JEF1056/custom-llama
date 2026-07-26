#!/usr/bin/env bash
#
# Deploy the MLX VLM server to multiple MacBooks (ml-2, ml-3, ml-4).
# Requires passwordless SSH to each host and HF_TOKEN available locally.
# All servers use the same hardcoded port (8080).
#
# Usage:
#   bash mac/deploy.sh                          # deploy to all hosts
#   bash mac/deploy.sh ml-2 ml-3                # deploy to specific hosts
#
# Env vars:
#   HF_TOKEN          — HuggingFace token
#   MLX_KV_BITS       — KV cache quantization bits (default: 4)
#   MLX_MAX_KV_SIZE   — max KV cache size (default: 229376)
#   CUSTOM_LLAMA_REF  — branch/commit to deploy from (default: hosting)
#
set -euo pipefail

# ---- Hosts ------------------------------------------------------------------
ALL_HOSTS=(ml-2 ml-3 ml-4)
HOSTS=("${@:-${ALL_HOSTS[@]}}")

# ---- Enable auto-login flag -------------------------------------------------
ENABLE_AUTO_LOGIN=${ENABLE_AUTO_LOGIN:-1}

# ---- Config -----------------------------------------------------------------
MLX_PORT="8080"
CUSTOM_LLAMA_REF=${CUSTOM_LLAMA_REF:-hosting}
MLX_KV_BITS=${MLX_KV_BITS:-4}
MLX_MAX_KV_SIZE=${MLX_MAX_KV_SIZE:-229376}

# ---- Colors -----------------------------------------------------------------
green()  { printf '\033[1;32m%s\033[0m' "$*"; }
red()    { printf '\033[1;31m%s\033[0m' "$*" >&2; }
yellow() { printf '\033[1;33m%s\033[0m' "$*"; }

# ---- Pre-flight: SSH connectivity -------------------------------------------
for host in "${HOSTS[@]}"; do
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" &>/dev/null; then
        red "ERROR: Cannot reach $host via SSH. Check DNS/SSH config."
        exit 1
    fi
done

green "Pre-flight OK: all hosts reachable\n"

# ---- Enable auto-login on each host -----------------------------------------
if [[ "$ENABLE_AUTO_LOGIN" -eq 1 ]]; then
    yellow "Enabling auto-login on all hosts..."
    
    for host in "${HOSTS[@]}"; do
        yellow "  $host: Setting auto-login..."
        ssh "$host" <<SSH_EOF
# Get the short username from the primary admin account
USERNAME=\$(defaults read /Library/Preferences/com.apple.loginwindow autoLoginUser)
if [[ -n "\$USERNAME" ]]; then
    # Enable auto-login with delayed prompt
    sudo defaults write /Library/Preferences/com.apple.loginwindow autoLoginUser -string "\$USERNAME"
    sudo defaults write /Library/Preferences/com.apple.loginwindow RetriesUntilTimeout -int 0
    echo "  $host: Auto-login enabled for user \$USERNAME"
else
    echo "  $host: Could not determine username, skipping auto-login"
fi
SSH_EOF
    done
    
    green "  Auto-login configured on all hosts\n"
fi

# ---- Deploy to each host ----------------------------------------------------
FAILED=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

for host in "${HOSTS[@]}"; do
    LABEL="com.custom-llama.qwen36-mlx.${host}"

    yellow "Deploying to $host (port=$MLX_PORT, label=$LABEL)..."

    # Create a temporary install wrapper for this host
    TEMP_INSTALL=$(mktemp)
    cat > "$TEMP_INSTALL" <<'INNER_EOF'
export HF_TOKEN='${HF_TOKEN:-}'
export MLX_PORT='${MLX_PORT}'
export MLX_KV_BITS='${MLX_KV_BITS}'
export MLX_MAX_KV_SIZE='${MLX_MAX_KV_SIZE}'
export CUSTOM_LLAMA_REF='${CUSTOM_LLAMA_REF}'
export LABEL='${LABEL}'
INNER_EOF

    # Append the actual install script
    cat "$REPO_ROOT/mac/install.sh" >> "$TEMP_INSTALL"

    # Transfer and run
    if scp "$TEMP_INSTALL" "$host:/tmp/deploy-install.sh" && \
       ssh "$host" "bash /tmp/deploy-install.sh" 2>&1 | tee "/tmp/deploy-${host}.log"; then
        green "  $host: SUCCESS (http://localhost:${MLX_PORT}/v1)\n"
    else
        red "  $host: FAILED (see /tmp/deploy-${host}.log)\n"
        FAILED+=("$host")
    fi

    rm -f "$TEMP_INSTALL" "/tmp/deploy-${host}.log"
done

# ---- Summary -----------------------------------------------------------------
echo "==========================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
    green "All hosts deployed successfully."
else
    red "Failed hosts: ${FAILED[*]}"
fi
echo "==========================================="
if [[ "$ENABLE_AUTO_LOGIN" -eq 1 ]]; then
    green "Auto-login enabled: servers will start at login"
fi
echo ""
echo "Ports:"
for host in "${HOSTS[@]}"; do
    if [[ ! " ${FAILED[*]:-} " =~ " ${host} " ]]; then
        green "  ${host}: http://localhost:${MLX_PORT}/v1"
    else
        red "  ${host}: http://localhost:${MLX_PORT}/v1 (deploy failed)"
    fi
done
