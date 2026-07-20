#!/usr/bin/env bash
#
# Phase 2 step 4 / Phase 4: quantize the merged BF16 GGUF with ik_llama's own
# llama-quantize using the "262K-Balanced" recipe (docs/iqllama-migration-plan.md
# section 4c): edge experts iq4_ks, middle experts iq3_k, shared expert q8_0,
# attention iq5_ks, router q8_0, token_embd iq4_ks, output q6_K.
#
# Expects the BF16 GGUF and imatrix to already be present (produced by
# download-source-gguf.sh, which fetches imatrix_unsloth.gguf_file alongside
# the BF16 shards).  Run prepare-weights.sh to do all steps in sequence, or
# set SRC_DIR/BF16_GGUF/IMATRIX/OUT_GGUF to override defaults.
#
# The MTP block (blk.40.*) FFN expert weights (ffn_gate/up/down_exps) are
# quantized to q8_0 for CUDA graph compatibility: BF16 weights use a generic
# fallback in ggml_cuda_moe_up_gate_unary that calls cudaStreamSynchronize
# (forbidden during CUDA graph capture), making ~65% of all decode calls
# graph-incompatible. q8_0 uses the fast quantized TG path which is fully
# graph-safe. attn+norm+router within blk.40 are kept at BF16 (small, and
# the attention/router ops are graph-compatible regardless of weight type).
# The MTP output tensor (blk.40.nextn.eh_proj.weight) uses q8_0 at load time
# via --mtp-requantize-output-tensor (wired via MTP_REQUANTIZE_OUTPUT_TYPE in
# docker/.env), consistent with real-hardware A/B testing showing q8_0 faster
# than bf16 for that tensor (~131.5 vs ~125.1 avg tok/s). The vision tower
# (mmproj-BF16.gguf) is a separate GGUF and is never quantized - copied as-is.
#
# VERIFIED against the real GGUF (unsloth/Qwen3.6-35B-A3B-MTP-GGUF BF16 shards,
# see docs/qwen36-bench-results.md): the --custom-q regexes below match real
# tensor names confirmed via direct GGUF header inspection:
#   blk.N.ffn_(gate|up|down)_exps.weight   (routed experts)
#   blk.N.ffn_(gate|up|down)_shexp.weight  (shared expert)
#   blk.N.attn_{q,k,v,output}.weight       (10 full-attention layers only)
#   blk.N.ffn_gate_inp.weight              (router)
#
# The 30 DeltaNet (linear-attention) layers have no attn_{q,k,v,output} tensors;
# they use ssm_* tensors (ssm_a, ssm_conv1d, ssm_dt, ssm_alpha, ssm_beta,
# ssm_norm) which fall through to BASE_TYPE (iq4_ks).
set -euo pipefail

SRC_DIR=${SRC_DIR:-/models/qwen36-src}
BF16_GGUF=${BF16_GGUF:-$SRC_DIR/qwen36-bf16.gguf}
IMATRIX=${IMATRIX:-$SRC_DIR/imatrix_unsloth.gguf_file}
OUT_GGUF=${OUT_GGUF:-/models/qwen36-262k-balanced.gguf}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}
# Base/default type for any tensor not matched by a more specific rule below.
BASE_TYPE=${BASE_TYPE:-iq4_ks}
# Layer ranges (0-indexed, 40 total layers): edge = most sensitive, kept a tier
# higher; middle = sparse bulk, the main quality/size trade-off.
EDGE_RANGE=${EDGE_RANGE:-'([0-4]|3[5-9])'}
MIDDLE_RANGE=${MIDDLE_RANGE:-'([5-9]|[12][0-9]|3[0-4])'}

if [[ ! -f "$BF16_GGUF" ]]; then
    echo "[quantize] ERROR: $BF16_GGUF not found; run download-source-gguf.sh first." >&2
    exit 1
fi

if [[ ! -f "$IMATRIX" ]]; then
    echo "[quantize] ERROR: $IMATRIX not found; run download-source-gguf.sh first" \
         "(it fetches imatrix_unsloth.gguf_file alongside the BF16 shards)." >&2
    exit 1
fi

echo "[quantize] $BF16_GGUF -> $OUT_GGUF (262K-Balanced recipe)"
echo "[quantize] imatrix: $IMATRIX"
"$LLAMA_BIN_DIR/llama-quantize" \
    --imatrix "$IMATRIX" \
    --custom-q "blk\.${EDGE_RANGE}\..*ffn_(gate|up|down)_exps\.weight=iq4_ks" \
    --custom-q "blk\.${MIDDLE_RANGE}\..*ffn_(gate|up|down)_exps\.weight=iq3_k" \
    --custom-q ".*ffn_(gate|up|down)_shexp\.weight=q8_0" \
    --custom-q "blk\.40\.attn_.*\.weight=bf16" \
    --custom-q "blk\.40\.ffn_(gate|up|down)_exps\.weight=q8_0" \
    --custom-q "blk\.40\.ffn_.*\.weight=bf16" \
    --custom-q "blk\.40\.nextn\..*\.weight=bf16" \
    --attn-q-type iq5_ks --attn-k-type iq5_ks --attn-v-type iq5_ks --attn-output-type iq5_ks \
    --ffn-gate-inp-type q8_0 \
    --token-embedding-type iq4_ks \
    --output-tensor-type q6_K \
    "$BF16_GGUF" "$OUT_GGUF" "$BASE_TYPE"

echo "[quantize] done: $OUT_GGUF"
