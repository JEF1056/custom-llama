#!/usr/bin/env bash
#
# Phase 2 step 4 / Phase 4: quantize the merged BF16 GGUF with ik_llama's own
# llama-quantize using the "262K-Balanced" recipe (docs/iqllama-migration-plan.md
# section 4c): edge experts iq4_ks, middle experts iq3_k, shared expert q8_0,
# attention iq5_ks, router q8_0, token_embd iq4_ks, output q6_K.
#
# The MTP block (blk.40.*, confirmed present via unsloth/Qwen3.6-35B-A3B-MTP-GGUF
# - see docs/qwen36-bench-results.md) is kept at BF16, same policy as the
# vision tower, EXCEPT the MTP output tensor (blk.40.nextn.eh_proj.weight):
# real-hardware A/B testing (docs/qwen36-bench-results.md) measured q8_0 vs
# bf16 for that specific tensor across 3 runs each and found q8_0 consistently
# faster (~131.5 avg tok/s vs ~125.1 for bf16) with no meaningful draft-
# acceptance-rate improvement from the extra precision - so it's the one part
# of the MTP block we do quantize. The vision tower (mmproj-BF16.gguf) is a
# separate GGUF file entirely and is never quantized - it's copied/symlinked
# as-is below, already BF16 from Unsloth.
#
# VERIFIED against the real GGUF (unsloth/Qwen3.6-35B-A3B-MTP-GGUF BF16 shards,
# see docs/qwen36-bench-results.md): the --custom-q regexes below match real
# tensor names confirmed via direct GGUF header inspection this session -
# `blk.N.ffn_(gate|up|down)_exps.weight` (routed experts), `blk.N.ffn_(gate|up|
# down)_shexp.weight` (shared expert), `blk.N.attn_{q,k,v,output}.weight` (the
# 10 full-attention layers only), `blk.N.ffn_gate_inp.weight` (router). These
# are no longer placeholders.
#
# The 30 DeltaNet (linear-attention) layers do NOT have attn_{q,k,v,output}
# tensors at all - they use a distinct `ssm_*` tensor family instead (ssm_a,
# ssm_conv1d.weight, ssm_dt(.bias), ssm_alpha.weight, ssm_beta.weight,
# ssm_norm.weight), confirmed via the same tensor dumps. The migration plan's
# section 4c recipe table does not call out a specific type for these -
# --attn-*-type only touches tensors that exist, i.e. only the 10 real
# attention layers - so ssm_* tensors fall through to BASE_TYPE (iq4_ks),
# the same tier as edge experts/token_embd. This is a deliberate choice given
# the plan's silence on this tensor class, not an oversight: iq4_ks is
# already the recipe's general "quality-conscious default" elsewhere, and
# ssm_* tensors are a comparatively small fraction of total weight (norms/
# gates/biases per layer, not the big expert matrices).
set -euo pipefail

SRC_DIR=${SRC_DIR:-/models/qwen36-src}
BF16_GGUF=${BF16_GGUF:-$SRC_DIR/qwen36-bf16.gguf}
IMATRIX=${IMATRIX:-$SRC_DIR/qwen36.imatrix}
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
    echo "[quantize] ERROR: $IMATRIX not found; run compute-imatrix.sh first" >&2
    echo "[quantize] (or point IMATRIX at $SRC_DIR/imatrix_unsloth.gguf_file)." >&2
    exit 1
fi

echo "[quantize] $BF16_GGUF -> $OUT_GGUF (262K-Balanced recipe)"
"$LLAMA_BIN_DIR/llama-quantize" \
    --imatrix "$IMATRIX" \
    --custom-q "blk\.${EDGE_RANGE}\..*ffn_(gate|up|down)_exps\.weight=iq4_ks" \
    --custom-q "blk\.${MIDDLE_RANGE}\..*ffn_(gate|up|down)_exps\.weight=iq3_k" \
    --custom-q ".*ffn_(gate|up|down)_shexp\.weight=q8_0" \
    --custom-q "blk\.40\.attn_.*\.weight=bf16" \
    --custom-q "blk\.40\.ffn_.*\.weight=bf16" \
    --custom-q "blk\.40\.nextn\..*\.weight=bf16" \
    --attn-q-type iq5_ks --attn-k-type iq5_ks --attn-v-type iq5_ks --attn-output-type iq5_ks \
    --ffn-gate-inp-type q8_0 \
    --token-embedding-type iq4_ks \
    --output-tensor-type q6_K \
    "$BF16_GGUF" "$OUT_GGUF" "$BASE_TYPE"

echo "[quantize] done: $OUT_GGUF"
echo "[quantize] MTP block (blk.40.*) kept at BF16 in the static GGUF; the"
echo "[quantize] eh_proj (MTP output) tensor is requantized to q8_0 at *load*"
echo "[quantize] *time* by llama-server's --mtp-requantize-output-tensor flag"
echo "[quantize] (wired via MTP_REQUANTIZE_OUTPUT_TYPE in docker/.env), not"
echo "[quantize] statically here - llama-quantize has no such flag."
echo "[quantize] copy/symlink $SRC_DIR/mmproj-BF16.gguf next to it (or requantize"
echo "[quantize] it separately) so entrypoint.sh's ENABLE_VISION=1 can find it."
echo "[quantize] (mmproj is already BF16 from Unsloth - left untouched either way.)"
