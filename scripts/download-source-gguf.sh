#!/usr/bin/env bash
#
# Phase 2 step 1-2: download the quantization SOURCE - llmfan46's abliterated
# ("Heretic") Qwen3.6-35B-A3B, published as a single-file BF16 GGUF with the
# native MTP layer PRESERVED + a matching BF16 mmproj. We quantize this with
# our own recipe (scripts/quantize.sh). The Heretic repo ships no imatrix, so
# Unsloth's imatrix for the same base architecture is reused by default (same
# qwen3_5_moe tensor names; see IMATRIX_REPO below).
#
# Requires: `hf` CLI (huggingface_hub[cli]) and ~75+ GB free disk.
set -euo pipefail

# Weights source: single-file BF16 GGUF with the native MTP (blk.40 nextn)
# layer preserved, plus a matching BF16 mmproj vision tower.
SRC_REPO=${SRC_REPO:-llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF}
BF16_SRC_FILE=${BF16_SRC_FILE:-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-BF16.gguf}
MMPROJ_SRC_FILE=${MMPROJ_SRC_FILE:-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf}
# The Heretic GGUF ships no imatrix, so reuse Unsloth's imatrix for the same
# base architecture (identical tensor names; activation stats are a reasonable
# proxy across the abliteration delta). Set IMATRIX_REPO="" to skip fetching an
# imatrix (then unset IMATRIX / drop --imatrix in quantize.sh to match).
IMATRIX_REPO=${IMATRIX_REPO:-unsloth/Qwen3.6-35B-A3B-MTP-GGUF}
IMATRIX_SRC_FILE=${IMATRIX_SRC_FILE:-imatrix_unsloth.gguf_file}
DEST_DIR=${DEST_DIR:-/models/qwen36-src}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}
GGUF_PY_DIR=${GGUF_PY_DIR:-/opt/iqllama/gguf-py}

mkdir -p "$DEST_DIR"

echo "[download] fetching BF16 GGUF + mmproj from $SRC_REPO -> $DEST_DIR"
# NOTE: `hf download REPO --include PAT1 PAT2 PAT3` is WRONG - the `hf` CLI
# treats any args after the first --include value as explicit filenames (not
# additional include patterns), silently disabling --include entirely (with
# only a warning, not an error) and downloading just those files. Repeat
# --include per pattern instead.
hf download "$SRC_REPO" \
    --include "$BF16_SRC_FILE" \
    --include "$MMPROJ_SRC_FILE" \
    --local-dir "$DEST_DIR"

if [[ -n "$IMATRIX_REPO" ]]; then
    echo "[download] fetching imatrix $IMATRIX_SRC_FILE from $IMATRIX_REPO -> $DEST_DIR"
    hf download "$IMATRIX_REPO" \
        --include "$IMATRIX_SRC_FILE" \
        --local-dir "$DEST_DIR"
fi

# Expose the downloaded files under the stable local names the rest of the
# pipeline (quantize.sh, prepare-weights.sh) expects, regardless of the
# upstream filename. This BF16 GGUF is a single file (not sharded), so no
# llama-gguf-split --merge step is needed.
MERGED="$DEST_DIR/qwen36-bf16.gguf"
if [[ ! -f "$DEST_DIR/$BF16_SRC_FILE" ]]; then
    echo "[download] ERROR: expected $DEST_DIR/$BF16_SRC_FILE not found" >&2
    exit 1
fi
ln -sf "$BF16_SRC_FILE" "$MERGED"
ln -sf "$MMPROJ_SRC_FILE" "$DEST_DIR/mmproj-BF16.gguf"

# Verify the native MTP layer really is present (the whole point of the
# "-Native-MTP-Preserved-" source): the recipe's blk.40 nextn/ffn rules and
# the server's --spec-type mtp both require it. Fail fast if it's missing.
# Uses gguf-py's GGUFReader (metadata only, no tensor-data read) since the
# image builds no llama-gguf-dump binary.
echo "[download] verifying native MTP (blk.40 / nextn) layer in the BF16 GGUF..."
if PYTHONPATH="$GGUF_PY_DIR" python3 - "$MERGED" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
mtp = [t.name for t in r.tensors if t.name.startswith("blk.40.") or "nextn" in t.name.lower()]
if mtp:
    print(f"[verify] MTP layer present: {len(mtp)} tensors (e.g. {mtp[0]})")
    sys.exit(0)
print("[verify] NO blk.40 / nextn tensors found", file=sys.stderr)
sys.exit(1)
PY
then
    echo "[download] OK: native MTP layer confirmed."
else
    echo "[download] ERROR: no MTP (blk.40 / nextn) tensors in $MERGED." >&2
    echo "[download] This source must retain the native MTP layer; aborting." >&2
    exit 1
fi

echo "[download] done. BF16 source: $MERGED"
echo "[download] mmproj: $DEST_DIR/mmproj-BF16.gguf"

# ik_llama.cpp's llama-quantize expects the legacy binary DAT imatrix format,
# but Unsloth ships a GGUF-format imatrix. Convert it using the bundled script.
IMATRIX_GGUF="$DEST_DIR/$IMATRIX_SRC_FILE"
IMATRIX_DAT="$DEST_DIR/imatrix_unsloth.dat"
if [[ -z "$IMATRIX_REPO" ]]; then
    echo "[download] IMATRIX_REPO empty; skipping imatrix (quantize.sh must run without --imatrix)."
elif [[ -f "$IMATRIX_DAT" ]]; then
    echo "[download] converted imatrix DAT already exists, skipping: $IMATRIX_DAT"
else
    echo "[download] converting GGUF imatrix -> DAT format for ik_llama.cpp..."
    # The converter lives in the ik_llama.cpp source tree, baked into the image
    # at /opt/iqllama/convert_imatrix_gguf_to_dat.py alongside gguf-py/.
    python3 /opt/iqllama/convert_imatrix_gguf_to_dat.py \
        "$IMATRIX_GGUF" --outfile "$IMATRIX_DAT"
    echo "[download] imatrix DAT: $IMATRIX_DAT"
fi
