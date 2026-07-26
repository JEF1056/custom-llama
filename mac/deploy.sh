#!/usr/bin/env bash
#
# Deploy the MLX VLM server to multiple MacBooks (ml-2, ml-3, ml-4).
# Requires passwordless SSH to each host and HF_TOKEN available locally.
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

# ---- Port assignment per host -----------------------------------------------
declare -A PORT_MAP=( [ml-2]=8081 [ml-3]=8082 [ml-4]=8083 )

# ---- Config -----------------------------------------------------------------
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

# ---- Deploy to each host ----------------------------------------------------
FAILED=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

for host in "${HOSTS[@]}"; do
    PORT=${PORT_MAP[$host]:-8081}
    LABEL="com.custom-llama.qwen36-mlx.${host}"

    yellow "Deploying to $host (port=$PORT, label=$LABEL)..."

    # Create a temporary install wrapper for this host
    TEMP_INSTALL=$(mktemp)
    cat > "$TEMP_INSTALL" <<'INNER_EOF'
export HF_TOKEN='${HF_TOKEN:-}'
export MLX_PORT='${PORT}'
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
        green "  $host: SUCCESS (http://localhost:${PORT}/v1)\n"
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
echo ""
echo "Ports:"
for host in "${HOSTS[@]}"; do
    PORT=${PORT_MAP[$host]:-8081}
    if [[ ! " ${FAILED[*]:-} " =~ " ${host} " ]]; then
        green "  ${host}: http://localhost:${PORT}/v1"
    else
        red "  ${host}: http://localhost:${PORT}/v1 (deploy failed)"
    fi
done
