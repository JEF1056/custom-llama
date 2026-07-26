#!/usr/bin/env bash
#
# One-line installer for the Qwen3.6-35B-A3B MLX VLM server on a MacBook
# (Apple Silicon). Downloads the BF16 GGUF from HF, converts to MLX FP16
# safetensors, applies K-quant quantization, then installs a LaunchAgent that
# starts the server at login and auto-restarts it on crash (KeepAlive).
#
# Usage (from a GitHub raw URL):
#   curl -fsSL https://raw.githubusercontent.com/JEF1056/custom-llama/main/mac/install.sh \
#     | bash
#
# Key env vars:
#   HF_TOKEN        — HuggingFace read token (required for model download)
#   MLX_PORT        — Server port (default: 8081)
#   MLX_KV_BITS     — KV cache quantization bits (default: 4)
#   MLX_MAX_KV_SIZE — Max KV cache size in tokens (default: 65536)
#   MODEL_PATH      — Custom model path (default: ~/.qwen/models/qwen36-mlx)
#   HF_REPO         — HF repo ID for the model (default: llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved)
#
# DSpark speculative decoding is intentionally OFF on Apple Silicon: at batch 1
# the verification pass does not amortize yet, so it is not a speedup here.
set -euo pipefail

# ---- Config (override via env) ----------------------------------------------
CUSTOM_LLAMA_REPO=${CUSTOM_LLAMA_REPO:-https://github.com/JEF1056/custom-llama.git}
CUSTOM_LLAMA_REF=${CUSTOM_LLAMA_REF:-hosting}
QWEN_HOME=${QWEN_HOME:-$HOME/.qwen}
MLX_PORT=${MLX_PORT:-8081}
MLX_KV_BITS=${MLX_KV_BITS:-4}
MLX_MAX_KV_SIZE=${MLX_MAX_KV_SIZE:-229376}
MODEL_PATH=${MODEL_PATH:-$QWEN_HOME/models/qwen36-mlx}
HF_REPO=${HF_REPO:-llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved}
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

# ---- Convert HF repo to MLX safetensors + quantize ----------------------------
MLX_DIR="$MODELS_DIR/qwen36-mlx"

if [[ ! -d "$MLX_DIR" ]]; then
    log "Converting HF repo to MLX safetensors and quantizing..."

    # Use mlx_vlm.convert to convert the model from Hugging Face to MLX safetensors
    # and quantize it in one step. This includes the vision tower parameters.
    python3 -m mlx_vlm.convert \
        --hf-path "$HF_REPO" \
        --mlx-path "$MLX_DIR" \
        --dtype bfloat16 \
        --quantize \
        --q-mode affine \
        --quant-predicate mixed_4_8 \
        ${HF_TOKEN:+--token "$HF_TOKEN"}

    # Clean up the HuggingFace cache to reclaim disk space (~10-30 GB).
    # The converted MLX safetensors are already saved; the HF cache is no
    # longer needed. Set HF_CACHE_CLEANUP=0 to skip this step.
    if [[ "${HF_CACHE_CLEANUP:-1}" == "1" ]]; then
        log "Cleaning up HuggingFace cache..."
        rm -rf "$HOME/.cache/huggingface" 2>/dev/null || true
        log "HF cache removed."
    fi

    log "Conversion and quantization complete: $MLX_DIR"
else
    log "MLX model already exists at $MLX_DIR, skipping conversion."
fi

QUANTIZED_DIR="$MLX_DIR"

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
MLX_MAX_KV_SIZE_ESC=$(sed_escape "$MLX_MAX_KV_SIZE")
VENV_DIR_ESC=$(sed_escape "$VENV_DIR")
VENV_BIN_ESC=$(sed_escape "$VENV_DIR/bin")

sed -e "s|__REPO__|$CL_DIR_ESC|g" \
    -e "s|__HOME__|$HOME_ESC|g" \
    -e "s|__MLX_PORT__|$MLX_PORT_ESC|g" \
    -e "s|__MODEL_PATH__|$MODEL_PATH_ESC|g" \
    -e "s|__MLX_MAX_KV_SIZE__|$MLX_MAX_KV_SIZE_ESC|g" \
    -e "s|__VENV_DIR__|$VENV_DIR_ESC|g" \
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
