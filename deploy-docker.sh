#!/bin/bash
# =============================================================================
# deploy-docker.sh - Docker Deployment & Cleanup Script for custom-llama
# Target Model: heretic-org/Qwen3.8-27B-heretic-ara (IQ4_KSS pure 4-bit)
# Host Target: ml-1-wsl (WSL2 / Linux Docker)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ALIAS="qwen3.8-27b-heretic-ara"
QUANT="IQ4_XS"

stop_and_clean() {
    echo "=== Stopping Docker services and cleaning up ==="
    docker compose down -v --remove-orphans 2>/dev/null || true
    docker stop llama-server llama-convert 2>/dev/null || true
    docker rm llama-server llama-convert 2>/dev/null || true
    echo "Services stopped and containers removed."
}

uninstall_all() {
    stop_and_clean
    echo "Removing docker networks and images for custom-llama..."
    docker network rm custom-llama_llama-net 2>/dev/null || true
    docker rmi llama-server:latest llama-convert:latest 2>/dev/null || true

    if [ "$1" == "--purge-models" ]; then
        echo "Purging models directory..."
        rm -rf ./models/*
    fi
    echo "=== Docker Cleanup Complete ==="
}

install_and_start() {
    echo "=== Deploying custom-llama with $MODEL_ALIAS ($QUANT) ==="

    echo "Building Docker images..."
    docker compose build llama-convert llama-server

    echo "Downloading model $MODEL_ALIAS ($QUANT)..."
    docker compose run --rm llama-convert download "$MODEL_ALIAS" --quant "$QUANT"

    echo "Starting llama-server container..."
    docker compose up -d llama-server

    echo "=== Docker Deployment Complete ==="
}

case "$1" in
    --install)
        install_and_start
        ;;
    --uninstall|--clean)
        uninstall_all "$2"
        ;;
    *)
        echo "Usage: $0 {--install|--uninstall|--clean [--purge-models]}"
        exit 1
        ;;
esac
