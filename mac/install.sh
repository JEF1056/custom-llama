#!/usr/bin/env bash
#
# One-line installer for the Bonsai-27B MLX server on a MacBook
# (Apple Silicon). Installs the model, then a LaunchAgent that starts the
# server at login and auto-restarts it on crash (KeepAlive).
#
# Usage (from a GitHub raw URL):
#   curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
#     | BONSAI_TOKEN=hf_xxx bash
#
# Vision: enabled by default (ENABLE_VISION=1). On Apple Silicon that requires
# the ternary (2-bit) 27B MLX build (via mlx-vlm); the 1-bit MLX build is
# text-only, so pass ENABLE_VISION=0 for the leaner 1-bit text-only build.
#
# DSpark speculative decoding is intentionally OFF on Apple Silicon: at batch 1
# the verification pass does not amortize yet, so it is not a speedup here.
set -euo pipefail

# ---- Config (override via env) ----------------------------------------------
CUSTOM_LLAMA_REPO=${CUSTOM_LLAMA_REPO:-https://github.com/YOURUSER/custom-llama.git}
CUSTOM_LLAMA_REF=${CUSTOM_LLAMA_REF:-main}
DEMO_REPO=${DEMO_REPO:-https://github.com/PrismML-Eng/Bonsai-demo.git}
BONSAI_HOME=${BONSAI_HOME:-$HOME/.bonsai}
MLX_PORT=${MLX_PORT:-8081}

LABEL=com.custom-llama.bonsai-mlx
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
CL_DIR="$BONSAI_HOME/custom-llama"
DEMO_DIR="$BONSAI_HOME/Bonsai-demo"

log() { printf '\033[1;32m[bonsai]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[bonsai]\033[0m %s\n' "$*" >&2; }

# ---- Preconditions ----------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || { err "macOS only."; exit 1; }
[[ "$(uname -m)" == "arm64" ]]  || { err "Apple Silicon required for MLX."; exit 1; }

if [[ -z "${BONSAI_TOKEN:-}" ]]; then
    err "BONSAI_TOKEN is required (HF read token; the Bonsai-27B repos are private)."
    err "Re-run:  curl -fsSL <url>/mac/install.sh | BONSAI_TOKEN=hf_xxx bash"
    exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
    log "Installing Xcode Command Line Tools..."
    xcode-select --install || true
    err "Complete the Xcode CLT install dialog, then re-run this script."
    exit 1
fi

mkdir -p "$BONSAI_HOME" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

# ---- Vision family selection ------------------------------------------------
# Image input on Apple Silicon requires the ternary (2-bit) 27B MLX build served
# via mlx-vlm; the 1-bit MLX build is text-only. ENABLE_VISION=1 provisions the
# ternary model + mlx-vlm venv so vision works at runtime, and is baked into the
# LaunchAgent so run-mlx-server.sh serves it.
case "${ENABLE_VISION:-1}" in
    0|false|no|off) ENABLE_VISION=0 ;;
    *)              ENABLE_VISION=1 ;;
esac
if [[ "$ENABLE_VISION" == "1" ]]; then
    SETUP_FAMILY=ternary
    SETUP_MLX_VLM=1
    log "Vision enabled: provisioning the ternary 2-bit 27B MLX build + mlx-vlm (the 1-bit MLX build is text-only)."
else
    SETUP_FAMILY=bonsai
    SETUP_MLX_VLM=0
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

# ---- Fetch Bonsai-demo and run setup (MLX 27B) ------------------------------
if [[ ! -d "$DEMO_DIR/.git" ]]; then
    log "Cloning Bonsai-demo..."
    git clone "$DEMO_REPO" "$DEMO_DIR"
fi

log "Running Bonsai-demo setup (MLX, family=$SETUP_FAMILY 27B). Downloads several GB and builds the MLX fork..."
(
    cd "$DEMO_DIR"
    BONSAI_FAMILY="$SETUP_FAMILY" \
    BONSAI_MODEL=27B \
    BONSAI_MLX_VLM="$SETUP_MLX_VLM" \
    BONSAI_OPENWEBUI=0 \
    BONSAI_CODE_INTERPRETER=0 \
    BONSAI_TOKEN="$BONSAI_TOKEN" \
    ./setup.sh
)

# ---- Render + install the LaunchAgent ---------------------------------------
log "Installing LaunchAgent ($LABEL)..."
sed -e "s|__REPO__|$CL_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    -e "s|__MLX_PORT__|$MLX_PORT|g" \
    -e "s|__ENABLE_VISION__|$ENABLE_VISION|g" \
    "$CL_DIR/mac/com.custom-llama.bonsai-mlx.plist.template" > "$PLIST_DST"

# (Re)load cleanly.
launchctl bootout   "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable    "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL" || true

log "Done."
log "MLX server: http://localhost:$MLX_PORT/v1  (auto-starts at login, auto-restarts on crash)"
log "Logs: ~/Library/Logs/bonsai-mlx.out.log  and  ~/Library/Logs/bonsai-mlx.err.log"
log "Uninstall: bash $CL_DIR/mac/uninstall.sh"
