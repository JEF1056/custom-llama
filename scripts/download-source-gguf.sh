#!/usr/bin/env bash
#
# Phase 2 step 1-2: download the quantization SOURCE - trohrbaugh's Heretic
# Qwen3.8-27B with Arbitrary-Rank Ablation (ARA), in safetensors format.
# Converts to single-file BF16 GGUF (with native MTP preserved) for quantization.
# The Heretic repo ships no imatrix, so we skip imatrix and quantize without it.
#
# Requires: `hf` CLI (huggingface_hub[cli]), llama.cpp convert script, and
# ~150+ GB free disk (safetensors ~55 GB + converted GGUF).
set -euo pipefail

# Weights source: dense model in safetensors format (6 safetensors shards).
# 64 layers: 48 Gated DeltaNet + 16 full-attention layers + native MTP.
SRC_REPO=${SRC_REPO:-trohrbaugh/Qwen3.8-27B-heretic-ara}
# Download all safetensors shards + index + config
SAFETENSORS_PAT="model-*.safetensors"
INDEX_FILE="model.safetensors.index.json"
CONFIG_FILE="config.json"
# Merged BF16 GGUF output (produced by conversion step below).
BF16_GGUF_FILE="qwen38-bf16.gguf"
# MMProj vision tower (BF16 GGUF from unsloth's Qwen3.8-27B-GGUF repo).
MMPROJ_REPO=${MMPROJ_REPO:-unsloth/Qwen3.8-27B-GGUF}
MMPROJ_FILE=${MMPROJ_FILE:-mmproj-BF16.gguf}
# Destination for safetensors source files.
SRC_DIR=${SRC_DIR:-/models/qwen38-src}
# Destination for the converted BF16 GGUF (same dir).
DEST_DIR=${DEST_DIR:-/models/qwen38-src}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}
GGUF_PY_DIR=${GGUF_PY_DIR:-/opt/iqllama/gguf-py}
# imatrix: skip (Heretic repo has none; dense model uses pure IQ4_KSS).
IMATRIX_REPO=${IMATRIX_REPO:-}

mkdir -p "$SRC_DIR"

echo "[download] fetching safetensors from $SRC_REPO -> $SRC_DIR"
hf download "$SRC_REPO" \
    --include "$SAFETENSORS_PAT" \
    --include "$INDEX_FILE" \
    --include "$CONFIG_FILE" \
    --local-dir "$SRC_DIR"

# Download the MMProj vision tower from unsloth's GGUF repo
MMPROJ_PATH="$SRC_DIR/$MMPROJ_FILE"
if [[ ! -f "$MMPROJ_PATH" ]]; then
    echo "[download] fetching mmproj $MMPROJ_FILE from $MMPROJ_REPO -> $SRC_DIR"
    hf download "$MMPROJ_REPO" \
        --include "$MMPROJ_FILE" \
        --local-dir "$SRC_DIR"
else
    echo "[download] mmproj already exists, skipping: $MMPROJ_PATH"
fi

# Convert safetensors to BF16 GGUF using llama.cpp's convert_hf_to_gguf.py
BF16_GGUF="$SRC_DIR/$BF16_GGUF_FILE"
if [[ ! -f "$BF16_GGUF" ]]; then
    echo "[convert] safetensors -> BF16 GGUF: $SRC_DIR -> $BF16_GGUF"
    # Find the convert script in the ik_llama.cpp source tree
    CONVERT_SCRIPT=$(find "$LLAMA_BIN_DIR/../../.." -name "convert_hf_to_gguf.py" -type f 2>/dev/null | head -1)
    if [[ -z "$CONVERT_SCRIPT" ]]; then
        # Fallback: try ggml/bin/ path or scripts dir
        CONVERT_SCRIPT=$(find /opt -name "convert_hf_to_gguf.py" -type f 2>/dev/null | head -1)
    fi
    if [[ -z "$CONVERT_SCRIPT" ]]; then
        echo "[convert] ERROR: convert_hf_to_gguf.py not found in ik_llama.cpp source" >&2
        exit 1
    fi
    python3 "$CONVERT_SCRIPT" \
        "$SRC_DIR" \
        --outfile "$BF16_GGUF" \
        --outtype f16
    echo "[convert] done: $BF16_GGUF"
else
    echo "[convert] BF16 GGUF already exists, skipping: $BF16_GGUF"
fi

# Verify the native MTP layer is present (64-layer model, trailing layer blk.63).
echo "[download] verifying native MTP (blk.63 / nextn) layer in the BF16 GGUF..."
if PYTHONPATH="$GGUF_PY_DIR" python3 - "$BF16_GGUF" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
mtp = [t.name for t in r.tensors if "nextn" in t.name.lower() or t.name.startswith("blk.63.")]
if mtp:
    print(f"[verify] MTP layer present: {len(mtp)} tensors (e.g. {mtp[0]})")
    sys.exit(0)
print("[verify] NO nextn tensors found", file=sys.stderr)
sys.exit(1)
PY
then
    echo "[download] OK: native MTP layer confirmed."
else
    echo "[download] ERROR: no MTP (nextn) tensors in $BF16_GGUF." >&2
    echo "[download] This source must retain the native MTP layer; aborting." >&2
    exit 1
fi

echo "[download] done. BF16 source: $BF16_GGUF"
echo "[download] mmproj: $SRC_DIR/$MMPROJ_FILE"

# No imatrix for this model (dense, pure IQ4_KSS quantization).
echo "[download] imatrix: skipped (dense model uses pure IQ4_KSS)."
