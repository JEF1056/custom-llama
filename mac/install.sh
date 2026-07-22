#!/usr/bin/env bash
#
# One-line installer for the Qwen3.6-35B-A3B MLX VLM server on a MacBook
# (Apple Silicon). Downloads the BF16 GGUF from HF, converts to MLX FP16
# safetensors, applies K-quant quantization, then installs a LaunchAgent that
# starts the server at login and auto-restarts it on crash (KeepAlive).
#
# Usage (from a GitHub raw URL):
#   curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
#     | bash
#
# Key env vars:
#   HF_TOKEN        — HuggingFace read token (required for model download)
#   MLX_PORT        — Server port (default: 8081)
#   MLX_KV_BITS     — KV cache quantization bits (default: 4)
#   MODEL_PATH      — Custom model path (default: ~/.qwen/models/qwen36-mlx/quantized)
#   HF_REPO         — HF repo ID for the GGUF (default: llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF)
#
# DSpark speculative decoding is intentionally OFF on Apple Silicon: at batch 1
# the verification pass does not amortize yet, so it is not a speedup here.
set -euo pipefail

# ---- Config (override via env) ----------------------------------------------
CUSTOM_LLAMA_REPO=${CUSTOM_LLAMA_REPO:-https://github.com/YOURUSER/custom-llama.git}
CUSTOM_LLAMA_REF=${CUSTOM_LLAMA_REF:-main}
QWEN_HOME=${QWEN_HOME:-$HOME/.qwen}
MLX_PORT=${MLX_PORT:-8081}
MLX_KV_BITS=${MLX_KV_BITS:-4}
MODEL_PATH=${MODEL_PATH:-$QWEN_HOME/models/qwen36-mlx/quantized}
HF_REPO=${HF_REPO:-llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF}
HF_GGUF_FILE=${HF_GGUF_FILE:-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-bf16.gguf}
VENV_NAME=${VENV_NAME:-mlx-venv}

LABEL=com.custom-llama.qwen36-mlx
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
CL_DIR="$QWEN_HOME/custom-llama"
VENV_DIR="$QWEN_HOME/$VENV_NAME"
MODELS_DIR="$QWEN_HOME/models"

log() { printf '\033[1;32m[qwen36]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[qwen36]\033[0m %s\n' "$*" >&2; }

# ---- Preconditions ----------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || { err "macOS only."; exit 1; }
[[ "$(uname -m)" == "arm64" ]]  || { err "Apple Silicon required for MLX."; exit 1; }

if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Using HF_TOKEN for authenticated downloads."
else
    log "No HF_TOKEN set; using anonymous downloads (public repos only)."
fi

# ---- Xcode CLT check --------------------------------------------------------
if ! xcode-select -p >/dev/null 2>&1; then
    log "Installing Xcode Command Line Tools..."
    xcode-select --install || true
    err "Complete the Xcode CLT install dialog, then re-run this script."
    exit 1
fi

# ---- Create directories ------------------------------------------------------
mkdir -p "$QWEN_HOME" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs" "$MODELS_DIR"

# ---- Python + venv setup ----------------------------------------------------
if ! command -v python3 &>/dev/null; then
    err "python3 not found. Install Python 3.11+ before running."
    exit 1
fi

PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Python version: $PYTHON_VER"

if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip first, then install MLX stack
log "Upgrading pip..."
pip install --upgrade pip setuptools wheel

log "Installing MLX stack (mlx-lm[torch], mlx-vlm[torch])..."
pip install "mlx-lm[torch]" "mlx-vlm[torch]"

# Pin MLX version
log "Pinning mlx==0.31.2..."
pip install "mlx==0.31.2"

log "MLX stack installed successfully."

# ---- Download BF16 GGUF from HuggingFace ------------------------------------
GGUF_DST="$MODELS_DIR/$HF_GGUF_FILE"

if [[ ! -f "$GGUF_DST" ]]; then
    log "Downloading BF16 GGUF from HF repo: $HF_REPO"
    log "File: $HF_GGUF_FILE"

    # Use hf CLI to download the GGUF file
    if ! command -v hf &>/dev/null; then
        log "Installing hf CLI..."
        pip install huggingface-hub
    fi

    log "Downloading model files..."
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf download \
            --token "$HF_TOKEN" \
            "$HF_REPO" \
            "$HF_GGUF_FILE" \
            --local-dir "$MODELS_DIR"
    else
        hf download \
            "$HF_REPO" \
            "$HF_GGUF_FILE" \
            --local-dir "$MODELS_DIR"
    fi

    if [[ ! -f "$GGUF_DST" ]]; then
        err "Download failed: $GGUF_DST not found."
        exit 1
    fi
    log "Downloaded: $GGUF_DST"
else
    log "GGUF already exists at $GGUF_DST, skipping download."
fi

# ---- Convert BF16 GGUF to MLX FP16 safetensors --------------------------------
MLX_FP16_DIR="$MODELS_DIR/qwen36-mlx"

if [[ ! -d "$MLX_FP16_DIR" ]]; then
    log "Converting GGUF to MLX FP16 safetensors..."
    mkdir -p "$MLX_FP16_DIR"

    python3 -m mlx_lm.convert \
        --model "$GGUF_DST" \
        --mlx-path "$MLX_FP16_DIR"

    log "Conversion complete: $MLX_FP16_DIR"
else
    log "MLX FP16 safetensors already exist at $MLX_FP16_DIR, skipping conversion."
fi

# ---- Quantize with custom affine quantization ----------------------------------
# Uses mlx-lm's built-in affine quantization with a custom preset matching
# the K-quant recipe: edge experts Q4, middle experts Q3, attention Q5,
# shared expert Q8, router/embed/lm_head at higher precision.
QUANTIZED_DIR="$MODEL_PATH"

if [[ ! -d "$QUANTIZED_DIR" ]]; then
    log "Running custom affine quantization..."

    # Clone custom-llama repo if not present (for the quantize script)
    if [[ ! -d "$CL_DIR" ]]; then
        log "Cloning custom-llama repo..."
        git clone "$CUSTOM_LLAMA_REPO" "$CL_DIR"
        git -C "$CL_DIR" checkout "$CUSTOM_LLAMA_REF" 2>/dev/null || true
    fi

    # Run quantization using mlx-lm's built-in quantize with custom preset
    python3 "$CL_DIR/scripts/quantize-mlx.sh" \
        --input "$MLX_FP16_DIR" \
        --output "$QUANTIZED_DIR" \
        --kv-bits "$MLX_KV_BITS"

    log "Quantization complete: $QUANTIZED_DIR"
else
    log "Quantized model already exists at $QUANTIZED_DIR, skipping quantization."
fi

# ---- Fetch our wrappers + plist (this repo) ---------------------------------
if [[ -d "$CL_DIR/.git" ]]; then
    log "Updating custom-llama repo..."
    git -C "$CL_DIR" fetch --depth 1 origin "$CUSTOM_LLAMA_REF" 2>/dev/null || true
    git -C "$CL_DIR" checkout "$CUSTOM_LLAMA_REF" 2>/dev/null || true
    git -C "$CL_DIR" pull --ff-only 2>/dev/null || true
else
    log "Cloning custom-llama repo..."
    git clone "$CUSTOM_LLAMA_REPO" "$CL_DIR"
    git -C "$CL_DIR" checkout "$CUSTOM_LLAMA_REF" 2>/dev/null || true
fi

# ---- Render + install the LaunchAgent ---------------------------------------
log "Installing LaunchAgent ($LABEL)..."
# Escape & and \ for sed safety (paths may contain these characters)
sed_escape() { printf '%s\n' "$1" | sed 's/[&\\/|]/\\&/g'; }
CL_DIR_ESC=$(sed_escape "$CL_DIR")
HOME_ESC=$(sed_escape "$HOME")
MLX_PORT_ESC=$(sed_escape "$MLX_PORT")
MODEL_PATH_ESC=$(sed_escape "$MODEL_PATH")
VENV_BIN_ESC=$(sed_escape "$VENV_DIR/bin")

sed -e "s|__REPO__|$CL_DIR_ESC|g" \
    -e "s|__HOME__|$HOME_ESC|g" \
    -e "s|__MLX_PORT__|$MLX_PORT_ESC|g" \
    -e "s|__MODEL_PATH__|$MODEL_PATH_ESC|g" \
    -e "s|__VENV_BIN__|$VENV_BIN_ESC|g" \
    "$CL_DIR/mac/com.custom-llama.qwen36-mlx.plist.template" > "$PLIST_DST"

# (Re)load cleanly.
launchctl bootout   "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable    "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL" || true

log "Done."
log "MLX server: http://localhost:$MLX_PORT/v1  (auto-starts at login, auto-restarts on crash)"
log "Logs: ~/Library/Logs/qwen36-mlx.out.log  and  ~/Library/Logs/qwen36-mlx.err.log"
log "Model: $QUANTIZED_DIR"
log "Uninstall: bash $CL_DIR/mac/uninstall.sh"
