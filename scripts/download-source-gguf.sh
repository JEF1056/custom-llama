#!/usr/bin/env bash
#
# Download source weights, convert/extract components idempotently, and quantize.
# Designed so re-running skips already-created artifacts unless missing.
#
set -euo pipefail

HF_REPO=${HF_REPO:-"trohrbaugh/Qwen3.8-27B-heretic-ara"}
IMATRIX_REPO=${IMATRIX_REPO:-"mradermacher/Qwen3.8-27B-heretic-ara-i1-GGUF"}
IMATRIX_SRC_FILE=${IMATRIX_SRC_FILE:-"Qwen3.8-27B-heretic-ara.imatrix.gguf"}
GGUF_REPO=${GGUF_REPO:-"mradermacher/Qwen3.8-27B-heretic-ara-i1-GGUF"}
GGUF_SRC_FILE=${GGUF_SRC_FILE:-"Qwen3.8-27B-heretic-ara.Q8_0.gguf"}
DEST_DIR=${DEST_DIR:-"/models/qwen38-src"}
GGUF_PY_DIR=${GGUF_PY_DIR:-"/opt/iqllama/gguf-py"}

mkdir -p "$DEST_DIR"

# 1. Fetch & convert imatrix (skip if exists)
IMATRIX_DAT="$DEST_DIR/Qwen3.8-27B-heretic-ara.imatrix.dat"
if [[ -f "$IMATRIX_DAT" && -s "$IMATRIX_DAT" ]]; then
    echo "[download] Imatrix DAT already up to date: $IMATRIX_DAT"
else
    echo "[download] Fetching imatrix ($IMATRIX_SRC_FILE) from $IMATRIX_REPO..."
    hf download "$IMATRIX_REPO" \
        --include "$IMATRIX_SRC_FILE" \
        --local-dir "$DEST_DIR"

    echo "[download] Converting GGUF imatrix -> DAT format..."
    python3 /opt/iqllama/convert_imatrix_gguf_to_dat.py \
        "$DEST_DIR/$IMATRIX_SRC_FILE" --outfile "$IMATRIX_DAT"
    echo "[download] Imatrix DAT ready: $IMATRIX_DAT"
fi

# 2. Source language model GGUF (skip if exists)
SRC_GGUF="$DEST_DIR/$GGUF_SRC_FILE"
if [[ -f "$SRC_GGUF" && -s "$SRC_GGUF" ]]; then
    echo "[download] Source GGUF already up to date: $SRC_GGUF"
else
    echo "[download] Downloading source GGUF ($GGUF_SRC_FILE) from $GGUF_REPO..."
    hf download "$GGUF_REPO" \
        --include "$GGUF_SRC_FILE" \
        --local-dir "$DEST_DIR"
fi

# 3. Vision Tower / Multimodal Projector extraction (skip if exists)
MMPROJ_BF16="$DEST_DIR/mmproj-BF16.gguf"
if [[ -f "$MMPROJ_BF16" && -s "$MMPROJ_BF16" ]]; then
    echo "[download] Vision mmproj GGUF already up to date: $MMPROJ_BF16"
else
    echo "[download] Extracting vision tower from safetensors..."
    HF_DIR="$DEST_DIR/hf_safetensors"
    mkdir -p "$HF_DIR"
    
    # Download vision/config files only
    hf download "$HF_REPO" \
        --include "config.json" \
        --include "model.safetensors.index.json" \
        --include "model-00001-of-00006.safetensors" \
        --include "model-00002-of-00006.safetensors" \
        --include "model-00003-of-00006.safetensors" \
        --include "model-00004-of-00006.safetensors" \
        --include "model-00005-of-00006.safetensors" \
        --include "model-00006-of-00006.safetensors" \
        --local-dir "$HF_DIR"

    PYTHONPATH="$GGUF_PY_DIR" python3 /scripts/extract-mmproj.py \
        "$HF_DIR" \
        "$MMPROJ_BF16"
fi

echo "[download] All source components verified in $DEST_DIR."
