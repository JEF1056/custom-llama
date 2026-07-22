#!/usr/bin/env bash
#
# Orchestrates the full Phase 2 offline weights pipeline (docs/
# iqllama-migration-plan.md): download Unsloth's BF16 source + imatrix ->
# quantize with the "262K-Balanced" recipe. Meant to run as the `model-prep`
# compose service/profile, writing into the same /models volume the server
# reads from.
#
# Each step is idempotent-ish (download resumes; quantize overwrites its own
# output), so re-running after a --force-recreate of the prep container just
# picks up where the persistent volume left off.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/download-source-gguf.sh"

SRC_DIR=${SRC_DIR:-/models/qwen36-src}
# Use Unsloth's shipped imatrix, converted to ik_llama.cpp's binary DAT format
# (download-source-gguf.sh does the conversion automatically).
# Override by setting IMATRIX=/path/to/custom.imatrix before calling this script.
export IMATRIX="${IMATRIX:-$SRC_DIR/imatrix_unsloth.dat}"

"$SCRIPT_DIR/quantize.sh"

MMPROJ_FILE=${MMPROJ_FILE:-mmproj-Q8_0.gguf}
if [[ ! -f "/models/$MMPROJ_FILE" ]]; then
    if [[ "$MMPROJ_FILE" == "mmproj-BF16.gguf" ]]; then
        cp "$SRC_DIR/mmproj-BF16.gguf" "/models/$MMPROJ_FILE"
    else
        # Default: Q8_0-quantize the vision tower's BF16 weight tensors
        # (~32% smaller / ~276MiB less VRAM at load, no observed vision-
        # quality regression - see quantize-mmproj.py header for the
        # real-hardware validation notes). Set MMPROJ_FILE=mmproj-BF16.gguf
        # to skip this and use the unquantized vision tower instead.
        "$SCRIPT_DIR/quantize-mmproj.py" "$SRC_DIR/mmproj-BF16.gguf" "/models/$MMPROJ_FILE"
    fi
fi

echo "[prepare] done: ${OUT_GGUF:-/models/qwen36-262k-balanced.gguf}"
echo "[prepare] server MODEL_SOURCE=local will pick it up on next start/recreate."
