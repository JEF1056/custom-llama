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
# Use Unsloth's shipped imatrix (downloaded alongside the BF16 shards above).
# Override by setting IMATRIX=/path/to/custom.imatrix before calling this script.
export IMATRIX="${IMATRIX:-$SRC_DIR/imatrix_unsloth.gguf_file}"

"$SCRIPT_DIR/quantize.sh"

MMPROJ_FILE=${MMPROJ_FILE:-mmproj-BF16.gguf}
if [[ ! -f "/models/$MMPROJ_FILE" ]]; then
    cp "$SRC_DIR/mmproj-BF16.gguf" "/models/$MMPROJ_FILE"
fi

echo "[prepare] done: ${OUT_GGUF:-/models/qwen36-262k-balanced.gguf}"
echo "[prepare] server MODEL_SOURCE=local will pick it up on next start/recreate."
