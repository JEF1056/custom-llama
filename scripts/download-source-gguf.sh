#!/usr/bin/env bash
#
# Phase 2 step 1-2: download Unsloth's pre-converted Qwen3.6-35B-A3B GGUF
# (BF16 shards + mmproj + their imatrix) as the quantization SOURCE, since
# stock ik_llama.cpp's convert_hf_to_gguf.py has no Qwen3_5MoeForConditional
# Generation registration (see docs/iqllama-migration-plan.md section 0 item 4).
#
# Requires: `hf` CLI (huggingface_hub[cli]) and ~70+ GB free disk.
set -euo pipefail

SRC_REPO=${SRC_REPO:-unsloth/Qwen3.6-35B-A3B-MTP-GGUF}
DEST_DIR=${DEST_DIR:-/models/qwen36-src}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}

mkdir -p "$DEST_DIR"

echo "[download] fetching BF16 shards + mmproj + imatrix from $SRC_REPO -> $DEST_DIR"
# NOTE: `hf download REPO --include PAT1 PAT2 PAT3` is WRONG - the `hf` CLI
# treats any args after the first --include value as explicit filenames (not
# additional include patterns), silently disabling --include entirely (with
# only a warning, not an error) and downloading just those files. Repeat
# --include per pattern instead.
hf download "$SRC_REPO" \
    --include "BF16/*" \
    --include "mmproj-BF16.gguf" \
    --include "imatrix_unsloth.gguf_file" \
    --local-dir "$DEST_DIR"

# If the BF16 weight is split (multi-shard), merge into a single GGUF so
# llama-imatrix/llama-quantize don't need to special-case the split naming.
FIRST_SHARD=$(find "$DEST_DIR/BF16" -maxdepth 1 -name '*-00001-of-*.gguf' | head -n1 || true)
MERGED="$DEST_DIR/qwen36-bf16.gguf"
if [[ -n "$FIRST_SHARD" ]]; then
    echo "[download] merging split shards starting at $FIRST_SHARD -> $MERGED"
    "$LLAMA_BIN_DIR/llama-gguf-split" --merge "$FIRST_SHARD" "$MERGED"
else
    SINGLE=$(find "$DEST_DIR/BF16" -maxdepth 1 -name '*.gguf' | head -n1 || true)
    if [[ -z "$SINGLE" ]]; then
        echo "[download] ERROR: no BF16 GGUF found under $DEST_DIR/BF16" >&2
        exit 1
    fi
    echo "[download] single-file BF16 GGUF, no merge needed: $SINGLE"
    ln -sf "$SINGLE" "$MERGED"
fi

echo "[download] done. BF16 source: $MERGED"
echo "[download] mmproj: $DEST_DIR/mmproj-BF16.gguf"
echo "[download] Unsloth imatrix: $DEST_DIR/imatrix_unsloth.gguf_file"
