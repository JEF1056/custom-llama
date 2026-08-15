#!/usr/bin/env bash
#
# Health check for the MLX VLM server.
# Usage: bash mac/healthcheck.sh [host]
#   Without host: check localhost
#   With host: check via SSH (ml-2, ml-3, ml-4)
#
set -euo pipefail

HOST=${1:-localhost}
MLX_PORT=${MLX_PORT:-8080}

if [[ "$HOST" == "localhost" ]]; then
    URL="http://localhost:${MLX_PORT}/v1/models"
else
    URL="http://localhost:${MLX_PORT}/v1/models"
    SSH_CMD="ssh $HOST"
fi

if [[ "$HOST" == "localhost" ]]; then
    if curl -sf --max-time 10 "$URL" >/dev/null 2>&1; then
        echo "[OK] Server healthy on port $MLX_PORT"
        exit 0
    else
        echo "[FAIL] Server not responding on port $MLX_PORT"
        exit 1
    fi
else
    if $SSH_CMD curl -sf --max-time 10 "http://localhost:${MLX_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[OK] $HOST: Server healthy on port $MLX_PORT"
        exit 0
    else
        echo "[FAIL] $HOST: Server not responding on port $MLX_PORT"
        exit 1
    fi
fi
