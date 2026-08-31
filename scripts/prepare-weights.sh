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

SRC_DIR=${SRC_DIR:-/models/qwen38-src}
export IMATRIX="${IMATRIX:-$SRC_DIR/Qwen3.8-27B-heretic-ara.imatrix.dat}"

"$SCRIPT_DIR/quantize.sh"

MMPROJ_FILE=${MMPROJ_FILE:-mmproj-Q8_0.gguf}
if [[ -f "$SRC_DIR/mmproj-BF16.gguf" ]]; then
    if [[ "$MMPROJ_FILE" == "mmproj-BF16.gguf" ]]; then
        echo "[prepare] Copying $SRC_DIR/mmproj-BF16.gguf -> /models/$MMPROJ_FILE..."
        cp "$SRC_DIR/mmproj-BF16.gguf" "/models/$MMPROJ_FILE"
    else
        echo "[prepare] Quantizing vision tower from $SRC_DIR/mmproj-BF16.gguf -> /models/$MMPROJ_FILE (Q8_0)..."
        "$SCRIPT_DIR/quantize-mmproj.py" "$SRC_DIR/mmproj-BF16.gguf" "/models/$MMPROJ_FILE"
    fi
fi

echo "[prepare] done: ${OUT_GGUF:-/models/qwen38-27b-heretic-ara-iq4_kss.gguf}"
echo "[prepare] server MODEL_SOURCE=local will pick it up on next start/recreate."
