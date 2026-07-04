#!/bin/bash
# Kill whatever is on port 8000 at the very top of the script
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

SCRIPT_PATH="/Users/jfan/Documents/start-vllm.sh"
PLIST_PATH="/Users/jfan/Library/LaunchAgents/com.jfan.vllm-mlx.plist"

LOCAL_MODEL_PATH="/Users/jfan/Documents/Qwen3.6-35B-A3B-MLX-4bit"
MTP_SOURCE_PATH="/Users/jfan/Documents/Qwen3.6-35B-A3B-MTP-4bit"
LAUNCHER_SCRIPT="/Users/jfan/Documents/run_vllm_mlx.py"

install_startup() {
    echo "Setting up startup script for macOS boot..."
    
    # Create LaunchAgents directory if it doesn't exist
    mkdir -p "/Users/jfan/Library/LaunchAgents"
    
    # Create the plist content
    cat <<EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jfan.vllm-mlx</string>
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
    
    # Load the LaunchAgent
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    launchctl load "$PLIST_PATH"
    
    echo "Startup agent installed at $PLIST_PATH and registered successfully!"
}

uninstall_all() {
    echo "Uninstalling vllm-mlx startup agent and removing models..."
    
    # Unload and remove plist
    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        rm -f "$PLIST_PATH"
        echo "Removed launch agent: $PLIST_PATH"
    fi
    
    # Remove models
    if [ -d "$LOCAL_MODEL_PATH" ]; then
        rm -rf "$LOCAL_MODEL_PATH"
        echo "Removed local 4-bit model: $LOCAL_MODEL_PATH"
    fi
    if [ -d "$MTP_SOURCE_PATH" ]; then
        rm -rf "$MTP_SOURCE_PATH"
        echo "Removed local MTP draft model: $MTP_SOURCE_PATH"
    fi
    
    # Remove Hugging Face hub caches
    echo "Removing Hugging Face cache for models..."
    rm -rf "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B"
    rm -rf "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-MTP-4bit"
    
    # Remove SSD cache directory
    rm -rf "/Users/jfan/.cache/vllm-mlx/ssd_cache"
    
    # Remove benchmark configs
    rm -f "/Users/jfan/Documents/vllm_benchmark.json"
    rm -f "/Users/jfan/Documents/benchmark.py"
    echo "Removed benchmark files."
    
    echo "Uninstallation complete!"
}

# Ensure vllm-mlx is installed
if ! command -v vllm-mlx &> /dev/null; then
    echo "vllm-mlx not found, installing..."
    pip3 install vllm-mlx --break-system-packages
fi

if [ "$1" == "--install" ]; then
    install_startup
    exit 0
elif [ "$1" == "--uninstall" ]; then
    uninstall_all
    exit 0
elif [ "$1" == "--reset" ]; then
    echo "Resetting vllm-mlx configurations (deleting benchmark json)..."
    rm -f "/Users/jfan/Documents/vllm_benchmark.json"
    echo "Reset complete! Benchmark will run on the next startup."
    exit 0
fi

# 1. Check and convert main model to MLX 4-bit using mlx_vlm.convert if not already done
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "Local MLX 4-bit model not found. Converting and quantizing Qwen/Qwen3.6-35B-A3B using mlx_vlm..."
    mlx_vlm.convert --hf-path Qwen/Qwen3.6-35B-A3B --mlx-path "$LOCAL_MODEL_PATH" -q --q-bits 4 --trust-remote-code
fi

# 2. Download MTP weights repo if not present
if [ ! -d "$MTP_SOURCE_PATH" ]; then
    echo "MTP weights not found locally. Downloading mlx-community/Qwen3.6-35B-A3B-MTP-4bit..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mlx-community/Qwen3.6-35B-A3B-MTP-4bit', local_dir='$MTP_SOURCE_PATH')
"
fi

# 3. Copy MTP weights into mtp/ subdirectory inside model directory (avoids VLM strict-load crash)
if [ ! -f "$LOCAL_MODEL_PATH/mtp/weights.safetensors" ]; then
    echo "Injecting MTP weights into mtp/ subdirectory..."
    mkdir -p "$LOCAL_MODEL_PATH/mtp"
    cp "$MTP_SOURCE_PATH/model.safetensors" "$LOCAL_MODEL_PATH/mtp/weights.safetensors"
fi

# 4. Configure local model config.json for LLM loading and MTP injection
if [ -f "$LOCAL_MODEL_PATH/config.json" ]; then
    echo "Configuring model config.json..."
    python3 -c "
import json
with open('$LOCAL_MODEL_PATH/config.json', 'r') as f:
    d = json.load(f)
# Remove image_token_id to bypass the VLM class loader bug
if 'image_token_id' in d:
    del d['image_token_id']
# Enable MTP injection flag
d['num_nextn_predict_layers'] = 1
# Apply user custom context and RoPE scaling parameters
d['max_position_embeddings'] = 262144
d['rope_scaling'] = {
    'type': 'yarn',
    'factor': 8.0,
    'original_max_position_embeddings': 32768
}
with open('$LOCAL_MODEL_PATH/config.json', 'w') as f:
    json.dump(d, f, indent=4)
"
fi

# 5. Prefix weight keys with 'mtp.' inside mtp/weights.safetensors if not already done
python3 -c "
import mlx.core as mx
path = '$LOCAL_MODEL_PATH/mtp/weights.safetensors'
weights = mx.load(path)
needs_save = False
new_weights = {}
for k, v in weights.items():
    if not k.startswith('mtp.'):
        new_weights['mtp.' + k] = v
        needs_save = True
    else:
        new_weights[k] = v
if needs_save:
    mx.save_safetensors(path, new_weights)
    print('Prefixed mtp/weights.safetensors keys successfully!')
"

# Run benchmark if not yet configured
BENCHMARK_JSON="/Users/jfan/Documents/vllm_benchmark.json"
if [ ! -f "$BENCHMARK_JSON" ]; then
    echo "Benchmark configuration not found. Running benchmarks for optimal MTP tokens..."
    python3 /Users/jfan/Documents/benchmark.py
fi

# Load benchmark result
MTP_TOKENS=$(python3 -c "import json; print(json.load(open('$BENCHMARK_JSON'))['fastest_mtp_num_draft_tokens'])" 2>/dev/null || echo 1)
echo "Benchmark configuration loaded. Fastest MTP draft tokens: $MTP_TOKENS"

# Clean up port 8000 and 8002 to be absolutely sure
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :8002 | xargs kill -9 2>/dev/null || true
lsof -ti :8080 | xargs kill -9 2>/dev/null || true

# Set SSD cache path
SSD_CACHE_DIR="/Users/jfan/.cache/vllm-mlx/ssd_cache"
mkdir -p "$SSD_CACHE_DIR"

# Start backend server directly on port 8000 via our launcher wrapper, exposed to local network
echo "Starting vllm-mlx server directly on port 8000..."
exec python3 "$LAUNCHER_SCRIPT" serve "$LOCAL_MODEL_PATH" \
    --served-model-name "Qwen/Qwen3.6-35B-A3B" \
    --enable-mtp \
    --mtp-num-draft-tokens "$MTP_TOKENS" \
    --continuous-batching \
    --use-paged-cache \
    --enable-prefix-cache \
    --kv-cache-quantization \
    --kv-cache-quantization-bits 4 \
    --cache-memory-percent 10 \
    --prefill-batch-size 1 \
    --completion-batch-size 32 \
    --prefill-step-size 1024 \
    --mllm-prefill-step-size 1024 \
    --max-num-seqs 16 \
    --default-thinking-token-budget 2048 \
    --timeout 3600 \
    --ssd-cache-dir "$SSD_CACHE_DIR" \
    --ssd-cache-max-gb 50 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
