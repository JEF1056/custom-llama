#!/usr/bin/env bash
#
# Phase 2 step 4 / Phase 4: quantize the merged BF16 GGUF to pure IQ4_KSS.
#
# This is a dense model (27B, 64 layers: 48 Gated DeltaNet + 16 full-attention),
# not a MoE model, so quantization is straightforward: all tensors use iq4_kss
# as the base type. The MTP trailing layer (blk.63 nextn tensors) stays at BF16
# for CUDA graph compatibility during decode.
#
# Expects the BF16 GGUF to already be present (produced by
# download-source-gguf.sh). Run prepare-weights.sh to do all steps in sequence,
# or set SRC_DIR/BF16_GGUF/OUT_GGUF to override defaults.
#
# Layer layout (64 total, 0-indexed):
#   0-47:  Gated DeltaNet layers (ssm_* tensors, no attn_*)
#   48-63: Full-attention layers (attn_{q,k,v,output} tensors)
#   blk.63: Also has MTP nextn tensors (multi-token prediction)
set -euo pipefail

SRC_DIR=${SRC_DIR:-/models/qwen38-src}
BF16_GGUF=${BF16_GGUF:-$SRC_DIR/qwen38-bf16.gguf}
OUT_GGUF=${OUT_GGUF:-/models/qwen38-27b-heretic-ara-iq4_kss.gguf}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}
# Base/default type for any tensor not matched by a more specific rule.
BASE_TYPE=${BASE_TYPE:-iq4_kss}
# MTP trailing layer (blk.63): keep nextn tensors at BF16 for CUDA graph safety.
# The rest of blk.63 (attn+norm) stays at base type (iq4_kss).
MTP_LAYER=${MTP_LAYER:-63}

if [[ ! -f "$BF16_GGUF" ]]; then
    echo "[quantize] ERROR: $BF16_GGUF not found; run download-source-gguf.sh first." >&2
    exit 1
fi

echo "[quantize] $BF16_GGUF -> $OUT_GGUF (pure IQ4_KSS)"

"$LLAMA_BIN_DIR/llama-quantize" \
    --custom-q "blk\.${MTP_LAYER}\.nextn\..*\.weight=bf16" \
    --custom-q "blk\.${MTP_LAYER}\.ffn_(gate|up|down)_exps\.weight=bf16" \
    "$BF16_GGUF" "$OUT_GGUF" "$BASE_TYPE"

echo "[quantize] done: $OUT_GGUF"
