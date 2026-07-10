#!/bin/bash
# Kill whatever is on port 8000
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Resolve our own absolute path dynamically
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
PLIST_PATH="/Users/jfan/Library/LaunchAgents/com.jfan.mlx-vlm.plist"

LOCAL_MODEL_PATH="/Users/jfan/Documents/Qwen3.6-35B-A3B-MLX-4bit"
DRAFT_MODEL_PATH="z-lab/Qwen3.6-35B-A3B-DFlash"

install_startup() {
    echo "Setting up startup script for macOS boot..."

    mkdir -p "/Users/jfan/Library/LaunchAgents"

    cat <<EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jfan.mlx-vlm</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/osascript</string>
        <string>-e</string>
        <string>tell application "Terminal" to do script "$SCRIPT_PATH --run"</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

    chmod 644 "$PLIST_PATH"

    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    launchctl load "$PLIST_PATH"

    echo "Startup agent installed at $PLIST_PATH and registered successfully!"
}

uninstall_startup() {
    echo "Uninstalling mlx-vlm startup agent and removing models..."

    # Unload and remove plist
    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        rm -f "$PLIST_PATH"
        echo "Removed launch agent: $PLIST_PATH"
    fi

    # Remove local model
    if [ -d "$LOCAL_MODEL_PATH" ]; then
        rm -rf "$LOCAL_MODEL_PATH"
        echo "Removed local 4-bit model: $LOCAL_MODEL_PATH"
    fi

    # Remove Hugging Face hub caches
    echo "Removing Hugging Face cache for models..."
    rm -rf "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B"
    rm -rf "$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash"

    # Remove APC disk cache
    rm -rf "$HOME/.cache/mlx-vlm/caching"

    # Remove benchmark config
    rm -f "/Users/jfan/Documents/vllm_benchmark.json"
    echo "Removed benchmark files."

    echo "Uninstallation complete!"
}

# Handle flags
if [ "$1" == "--install" ]; then
    # Always reinstall and re-patch
    echo "Reinstalling mlx-vlm..."
    pip3 install -U mlx-vlm --break-system-packages

    PATCH_SCRIPT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/patch-model-alias.sh"
    if [ -f "$PATCH_SCRIPT" ]; then
        bash "$PATCH_SCRIPT"
    fi

    install_startup
    exit 0
elif [ "$1" == "--uninstall" ]; then
    uninstall_startup
    exit 0
elif [ "$1" == "--reset" ]; then
    echo "Resetting benchmark configuration..."
    rm -f "/Users/jfan/Documents/vllm_benchmark.json"
    echo "Reset complete! Benchmark will run on the next startup."
    exit 0
elif [ "$1" == "--run" ]; then
    shift
fi

# Auto-convert to MLX 4-bit if model not found locally
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "Local MLX 4-bit model not found. Converting and quantizing Qwen/Qwen3.6-35B-A3B..."
    mlx_vlm.convert --hf-path Qwen/Qwen3.6-35B-A3B --mlx-path "$LOCAL_MODEL_PATH" -q --q-bits 4 --trust-remote-code
fi

# Optimizations:
#   KV Cache Quantization — 4-bit TurboQuant (~4x KV reduction)
#   Continuous batching — auto-enabled when model is preloaded
#   Vision feature cache — auto-enabled on server
#   Speculative decoding — via --draft-model (DFlash, ~2-3x speedup)
# Note: APC is mutually exclusive with KV cache quantization, so we use TurboQuant.

# Start mlx-vlm server with all optimizations enabled
echo "Starting mlx-vlm server on port 8000..."
exec mlx_vlm.server \
    --model "$LOCAL_MODEL_PATH" \
    --draft-model "$DRAFT_MODEL_PATH" \
    --kv-bits 4 \
    --kv-quant-scheme turboquant \
    --enable-thinking \
    --thinking-budget 2048 \
    --thinking-start-token "<think>" \
    --thinking-end-token "</think>" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "qwen3.6-35b-a3b"
