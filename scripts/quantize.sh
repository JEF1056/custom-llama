#!/usr/bin/env bash
#
# Phase 2 step 4 / Phase 4: quantize the merged BF16 GGUF with ik_llama's own
# llama-quantize using the "262K-Balanced" recipe (docs/iqllama-migration-plan.md
# section 4c): edge experts iq4_kss, middle experts iq3_ks, shared expert q8_0,
# attention iq5_ks, router q8_0, token_embd iq4_kss, output q6_K.
#
# iq4_kss/iq3_ks replace iq4_ks/iq3_k (same tiers, ~4.0/~3.19 bpw vs ~4.25/
# ~3.44 bpw) for a smaller footprint at equivalent CUDA support: both are
# fully accelerated (mmq/mmvq/mmq_id, incl. the MoE-expert path) and are
# real quantized types, so they pass the same CUDA-graph-capture check as
# the types they replace (ggml_is_quantized(type) == true - unlike the
# BF16 MTP tensors below, which intentionally stay off the graph path).
# Not independently quality-validated against iq4_ks/iq3_k on this model -
# re-check output quality after this change.
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
# (mmproj-BF16.gguf) is a separate GGUF handled by scripts/quantize-mmproj.py
# (called from prepare-weights.sh), not this script - llama-quantize can't
# quantize it directly since it hard-requires a recognized LLM architecture
# and the mmproj file's general.architecture is "clip".
#
# Quantization SOURCE is llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-
# Preserved-GGUF (abliterated Qwen3.6, single-file BF16, native MTP blk.40
# preserved). Same qwen3_5_moe tensor layout as the previously-used
# unsloth/Qwen3.6-35B-A3B-MTP-GGUF, so the recipe below (incl. the blk.40
# nextn/ffn MTP rules) is unchanged. The --custom-q regexes match real tensor
# names confirmed via direct GGUF header inspection:
#   blk.N.ffn_(gate|up|down)_exps.weight   (routed experts)
#   blk.N.ffn_(gate|up|down)_shexp.weight  (shared expert)
#   blk.N.attn_{q,k,v,output}.weight       (10 full-attention layers only)
#   blk.N.ffn_gate_inp.weight              (router)
#
# The 30 DeltaNet (linear-attention) layers have no attn_{q,k,v,output} tensors;
# they use ssm_* tensors. The DeltaNet output projection (blk.N.ssm_out.weight,
# shape {value_dim, n_embd}) is held a tier higher at iq6_k (6.6 bpw) - it's a
# real, quality-relevant matrix analogous to attn_output. (There is no q6_ks
# type in ik_llama; iq6_k is the non-linear IQK 6-bit, consistent with the
# iq5_ks attention tier.) The remaining ssm_* tensors (ssm_a, ssm_conv1d,
# ssm_dt, ssm_alpha, ssm_beta, ssm_norm, ssm_beta_alpha) fall through to
# BASE_TYPE (iq4_kss).
set -euo pipefail

SRC_DIR=${SRC_DIR:-/models/qwen38-src}
SRC_GGUF=${SRC_GGUF:-$SRC_DIR/Qwen3.8-27B-heretic-ara.Q8_0.gguf}
IMATRIX=${IMATRIX:-$SRC_DIR/Qwen3.8-27B-heretic-ara.imatrix.dat}
OUT_GGUF=${OUT_GGUF:-/models/qwen38-27b-heretic-ara-iq4_kss.gguf}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}
QUANT_TYPE=${QUANT_TYPE:-iq4_kss}

if [[ ! -f "$SRC_GGUF" ]]; then
    # Check alternate filenames
    if [[ -f "$SRC_DIR/qwen38-bf16.gguf" ]]; then
        SRC_GGUF="$SRC_DIR/qwen38-bf16.gguf"
    elif [[ -f "/models/Qwen3.8-27B-heretic-ara.Q8_0.gguf" ]]; then
        SRC_GGUF="/models/Qwen3.8-27B-heretic-ara.Q8_0.gguf"
    else
        echo "[quantize] ERROR: Source GGUF ($SRC_GGUF) not found; run download-source-gguf.sh first." >&2
        exit 1
    fi
fi

IMATRIX_ARGS=()
if [[ -f "$IMATRIX" ]]; then
    echo "[quantize] using imatrix: $IMATRIX"
    IMATRIX_ARGS=(--imatrix "$IMATRIX")
elif [[ -f "$SRC_DIR/Qwen3.8-27B-heretic-ara.imatrix.gguf" ]]; then
    echo "[quantize] converting GGUF imatrix -> DAT format..."
    python3 /opt/iqllama/convert_imatrix_gguf_to_dat.py \
        "$SRC_DIR/Qwen3.8-27B-heretic-ara.imatrix.gguf" --outfile "$SRC_DIR/Qwen3.8-27B-heretic-ara.imatrix.dat"
    IMATRIX="$SRC_DIR/Qwen3.8-27B-heretic-ara.imatrix.dat"
    IMATRIX_ARGS=(--imatrix "$IMATRIX")
fi

if [[ -f "$OUT_GGUF" && -s "$OUT_GGUF" ]]; then
    echo "[quantize] Target quantized GGUF already exists: $OUT_GGUF (skipping quantize)"
else
    echo "[quantize] Quantizing $SRC_GGUF -> $OUT_GGUF (pure $QUANT_TYPE)"
    "$LLAMA_BIN_DIR/llama-quantize" \
        --allow-requantize \
        "${IMATRIX_ARGS[@]}" \
        "$SRC_GGUF" "$OUT_GGUF" "$QUANT_TYPE"
fi

echo "[quantize] done: $OUT_GGUF"
