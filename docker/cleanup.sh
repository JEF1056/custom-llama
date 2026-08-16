#!/usr/bin/env bash
#
# Clean up docker build cache, volumes, and images on ml-1-wsl
# Preserves mcp-search-server while purging stale llama containers/cache/images.
#
# Usage:
#   bash docker/cleanup.sh [--purge-all]
#
set -euo pipefail

PURGE_ALL=${1:-""}

echo "[docker-cleanup] Stopping stale llama containers..."
docker stop llama-server llama-convert server 2>/dev/null || true
docker rm llama-server llama-convert server 2>/dev/null || true

echo "[docker-cleanup] Pruning build cache..."
docker builder prune -af

echo "[docker-cleanup] Pruning dangling images..."
docker image prune -f

echo "[docker-cleanup] Pruning dangling volumes..."
docker volume prune -f

if [[ "$PURGE_ALL" == "--purge-all" ]]; then
    echo "[docker-cleanup] Removing old llama images (keeping mcp-search-server)..."
    docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep -v 'mcp-search-server' | awk '{print $2}' | xargs -r docker rmi -f 2>/dev/null || true
fi

echo "[docker-cleanup] Complete. Current docker status:"
docker ps -a
docker system df
