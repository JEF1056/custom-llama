#!/bin/bash
# =============================================================================
# deploy-mlx.sh - Generic MLX Model Server Manager for macOS
# Supports installing, converting, running, and uninstalling any HuggingFace model.
# =============================================================================

set -e

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Default Configuration
DEFAULT_HF_REPO="heretic-org/Qwen3.8-27B-heretic-ara"
DEFAULT_QUANT_MODE="mxfp4"
DEFAULT_PORT="8080"
DEFAULT_KV_BITS="4"

HF_REPO="${HF_REPO:-$DEFAULT_HF_REPO}"
QUANT_MODE="${QUANT_MODE:-$DEFAULT_QUANT_MODE}"
PORT="${PORT:-$DEFAULT_PORT}"
KV_BITS="${KV_BITS:-$DEFAULT_KV_BITS}"

# Derive model slug from HF repo (e.g. heretic-org/Qwen3.8-27B-heretic-ara -> Qwen3.8-27B-heretic-ara)
MODEL_SLUG="$(echo "$HF_REPO" | awk -F'/' '{print $NF}')"
HF_ORG="$(echo "$HF_REPO" | awk -F'/' '{if (NF>1) print $1; else print "none"}')"

MODEL_DIR="$HOME/.qwen/models/${MODEL_SLUG}-${QUANT_MODE}"
ACTIVE_SYMLINK="$HOME/active-mlx-model"
PLIST_LABEL="com.jfan.mlx-server"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"

stop_server_on_port() {
    local target_port="$1"
    echo "Stopping MLX server processes on port ${target_port}..."
    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
    fi
    lsof -ti :"${target_port}" | xargs kill -9 2>/dev/null || true
    pkill -f "mlx_vlm.server.*--port ${target_port}" 2>/dev/null || true
    echo "Processes stopped on port ${target_port}."
}

uninstall_model() {
    local target_repo="$1"
    local slug="$(echo "$target_repo" | awk -F'/' '{print $NF}')"
    local org="$(echo "$target_repo" | awk -F'/' '{if (NF>1) print $1; else print "*"}')"

    echo "=== Uninstalling Model: ${target_repo} ==="
    stop_server_on_port "$PORT"

    if [ -f "$PLIST_PATH" ]; then
        rm -f "$PLIST_PATH"
        echo "Removed LaunchAgent plist: $PLIST_PATH"
    fi

    # Remove model directory matching slug
    rm -rf "$HOME/.qwen/models/${slug}"* 2>/dev/null || true
    echo "Removed model directory for ${slug}"

    # Remove active symlink if pointing to this model
    if [ -L "$ACTIVE_SYMLINK" ]; then
        rm -f "$ACTIVE_SYMLINK"
    fi

    # Clean up Hugging Face hub cache dynamically
    if [ "$org" != "*" ]; then
        echo "Removing Hugging Face cache for models--${org}--${slug}..."
        rm -rf "$HOME/.cache/huggingface/hub/models--${org}--${slug}" 2>/dev/null || true
    fi

    echo "=== Model Uninstall Complete ==="
}

clean_all() {
    echo "=== Cleaning All MLX Deployments & LaunchAgents ==="
    pkill -f "mlx_vlm.server" 2>/dev/null || true
    lsof -ti :$PORT | xargs kill -9 2>/dev/null || true

    for plist in "$HOME"/Library/LaunchAgents/com.jfan.mlx-server*.plist "$HOME"/Library/LaunchAgents/com.custom-llama*.plist; do
        if [ -f "$plist" ]; then
            launchctl unload "$plist" 2>/dev/null || true
            rm -f "$plist"
            echo "Unloaded and removed plist: $plist"
        fi
    done

    rm -f "$ACTIVE_SYMLINK" 2>/dev/null || true
    echo "=== All Clean Complete ==="
}

convert_model() {
    if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
        echo "Model already converted at $MODEL_DIR"
    else
        echo "Converting $HF_REPO to $QUANT_MODE MLX model at $MODEL_DIR..."
        mkdir -p "$(dirname "$MODEL_DIR")"
        python3 -m mlx_vlm.convert \
            --hf-path "$HF_REPO" \
            --mlx-path "$MODEL_DIR" \
            -q \
            --q-mode "$QUANT_MODE" \
            --trust-remote-code
        echo "Conversion complete!"
    fi

    # Update active symlink
    ln -sfn "$MODEL_DIR" "$ACTIVE_SYMLINK"
    echo "Active symlink updated: $ACTIVE_SYMLINK -> $MODEL_DIR"
}

install_startup() {
    stop_server_on_port "$PORT"
    convert_model

    echo "Setting up LaunchAgent startup service for ${MODEL_SLUG} on port ${PORT}..."
    mkdir -p "$HOME/Library/LaunchAgents"

    PYTHON_BIN="$(which python3)"

    cat <<EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>-m</string>
        <string>mlx_vlm.server</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>${PORT}</string>
        <string>--model</string>
        <string>${MODEL_DIR}</string>
        <string>--kv-bits</string>
        <string>${KV_BITS}</string>
        <string>--kv-quant-scheme</string>
        <string>uniform</string>
        <string>--kv-group-size</string>
        <string>64</string>
        <string>--prefill-step-size</string>
        <string>1024</string>
        <string>--max-kv-size</string>
        <string>131072</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>APC_ENABLED</key>
        <string>1</string>
        <key>APC_NUM_BLOCKS</key>
        <string>16384</string>
        <key>APC_BLOCK_SIZE</key>
        <string>16</string>
        <key>APC_EXACT_CACHE_ENTRIES</key>
        <string>16</string>
        <key>APC_DISK_PATH</key>
        <string>${HOME}/.cache/mlx-vlm/caching</string>
        <key>APC_DISK_MAX_GB</key>
        <string>40</string>
        <key>APC_DISK_SHARD_MAX_BLOCKS</key>
        <string>1024</string>
        <key>APC_DISK_WORKERS</key>
        <string>4</string>
        <key>MLX_METAL_FAST_SYNCHRONIZATION</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${HOME}/.qwen/mlx-server.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/.qwen/mlx-server-err.log</string>
</dict>
</plist>
EOF

    chmod 644 "$PLIST_PATH"
    launchctl load "$PLIST_PATH"
    echo "LaunchAgent registered and loaded: $PLIST_PATH"
}

run_foreground() {
    stop_server_on_port "$PORT"
    convert_model
    export APC_ENABLED=1
    export APC_NUM_BLOCKS=16384
    export APC_BLOCK_SIZE=16
    export APC_EXACT_CACHE_ENTRIES=16
    export APC_DISK_PATH="${HOME}/.cache/mlx-vlm/caching"
    export APC_DISK_MAX_GB=40
    export APC_DISK_SHARD_MAX_BLOCKS=1024
    export APC_DISK_WORKERS=4
    export MLX_METAL_FAST_SYNCHRONIZATION=1
    echo "Starting mlx_vlm.server for ${MODEL_SLUG} on port $PORT with optimized hardware config..."
    exec python3 -m mlx_vlm.server \
        --host 0.0.0.0 \
        --port "$PORT" \
        --model "$MODEL_DIR" \
        --kv-bits "$KV_BITS" \
        --kv-quant-scheme uniform \
        --kv-group-size 64 \
        --prefill-step-size 1024 \
        --max-kv-size 131072
}

show_status() {
    echo "=== MLX Server Status ==="
    echo "Active Symlink : $(readlink "$ACTIVE_SYMLINK" 2>/dev/null || echo "None")"
    echo "LaunchAgent    : $(launchctl list | grep com.jfan.mlx-server || echo "Not loaded")"
    echo "Running Process: $(ps aux | grep -v grep | grep "mlx_vlm.server" || echo "None")"
}

case "$1" in
    --install)
        if [ -n "$2" ]; then HF_REPO="$2"; fi
        if [ -n "$3" ]; then QUANT_MODE="$3"; fi
        if [ -n "$4" ]; then PORT="$4"; fi
        install_startup
        ;;
    --uninstall)
        TARGET="${2:-$HF_REPO}"
        uninstall_model "$TARGET"
        ;;
    --clean-all)
        clean_all
        ;;
    --run)
        if [ -n "$2" ]; then HF_REPO="$2"; fi
        run_foreground
        ;;
    --status)
        show_status
        ;;
    *)
        echo "Usage: $0 {--install [HF_REPO] [QUANT_MODE] [PORT] | --uninstall [HF_REPO] | --clean-all | --run | --status}"
        exit 1
        ;;
esac
