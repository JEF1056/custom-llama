#!/usr/bin/env bash
#
# Orchestrates the full Phase 2 offline weights pipeline for Qwen3.8-27B-heretic-ara:
# download safetensors -> convert to BF16 GGUF -> quantize to IQ4_KSS.
# Quantizes the vision tower's mmproj to Q8_0 (~32% smaller VRAM at load).
# Meant to run as the `model-prep` compose service/profile, writing into the
# same /models volume the server reads from.
#
# Each step is idempotent-ish (download resumes; quantize overwrites its own
# output), so re-running after a --force-recreate of the prep container just
# picks up where the persistent volume left off.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/download-source-gguf.sh"

SRC_DIR=${SRC_DIR:-/models/qwen38-src}

"$SCRIPT_DIR/quantize.sh"

MMPROJ_FILE=${MMPROJ_FILE:-mmproj-Q8_0.gguf}
if [[ ! -f "/models/$MMPROJ_FILE" ]]; then
    MMPROJ_BF16="$SRC_DIR/mmproj-BF16.gguf"
    if [[ "$MMPROJ_FILE" == "mmproj-BF16.gguf" ]]; then
        cp "$MMPROJ_BF16" "/models/$MMPROJ_FILE"
    else
        # Q8_0-quantize the vision tower's BF16 weight tensors
        # (~32% smaller / ~276MiB less VRAM at load, no observed vision-quality regression).
        "$SCRIPT_DIR/quantize-mmproj.py" "$MMPROJ_BF16" "/models/$MMPROJ_FILE"
    fi
fi

echo "[prepare] done: ${OUT_GGUF:-/models/qwen38-27b-heretic-ara-iq4_kss.gguf}"
echo "[prepare] server MODEL_SOURCE=local will pick it up on next start/recreate."
