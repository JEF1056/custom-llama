#!/usr/bin/env bash
#
# Phase 2 step 4 / Phase 4: quantize the merged BF16 GGUF with ik_llama's own
# llama-quantize using the "262K-Balanced" recipe (docs/iqllama-migration-plan.md
# section 4c): edge experts iq4_ks, middle experts iq3_k, shared expert q8_0,
# attention iq5_ks, router q8_0, token_embd iq4_ks, output q6_K.
#
# If the imatrix file does not already exist, this script builds it automatically:
#   1. Downloads the turboderp/exllamav3 standard calibration corpus files
#      (wiki, c4, code, multilingual, technical, tiny) from GitHub into a
#      temp directory that is removed unconditionally when the script exits.
#   2. Runs llama-imatrix over the BF16 source GGUF to produce $IMATRIX.
#   3. Proceeds with quantization.
# The imatrix is kept after completion (small, reusable). Re-running this script
# a second time skips steps 1-2 since $IMATRIX will already exist.
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

# Resolve the scripts directory so BUNDLED_IMATRIX works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SRC_DIR=${SRC_DIR:-/models/qwen36-src}
BF16_GGUF=${BF16_GGUF:-$SRC_DIR/qwen36-bf16.gguf}
IMATRIX=${IMATRIX:-$SRC_DIR/qwen36.imatrix}
# Pre-computed imatrix committed to scripts/ (git-tracked binary, fast path).
# Set to empty string to force recompute: BUNDLED_IMATRIX= ./quantize.sh
# Tip: if the file exceeds ~50 MB consider tracking it with Git LFS.
BUNDLED_IMATRIX=${BUNDLED_IMATRIX:-$SCRIPT_DIR/qwen36-bf16.imatrix}
OUT_GGUF=${OUT_GGUF:-/models/qwen36-262k-balanced.gguf}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}
CHUNKS=${CHUNKS:-200}
NGL=${NGL:-999}
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

# ---------------------------------------------------------------------------
# Imatrix resolution order:
#   1. $IMATRIX already on disk (previous run or externally supplied).
#   2. $BUNDLED_IMATRIX — pre-computed file committed to the scripts/ dir.
#      Copy it into place so subsequent steps always find $IMATRIX.
#   3. Compute from scratch: download exllamav3 corpus, run llama-imatrix.
# ---------------------------------------------------------------------------
if [[ ! -f "$IMATRIX" ]]; then
    if [[ -f "$BUNDLED_IMATRIX" ]]; then
        echo "[quantize] using bundled imatrix: $BUNDLED_IMATRIX"
        mkdir -p "$(dirname "$IMATRIX")"
        cp "$BUNDLED_IMATRIX" "$IMATRIX"
    else
    echo "[quantize] imatrix not found — building corpus..."

    _CORPUS_TMPDIR=$(mktemp -d)
    trap 'echo "[quantize] cleaning up corpus tmpdir..."; rm -rf "$_CORPUS_TMPDIR"' EXIT
    export _CORPUS_TMPDIR

    # ------------------------------------------------------------------
    # Corpus assembly — three complementary sources:
    #
    #   1. bartowski calibration_datav5 (base, ~4 MB):
    #      General text, code/debug exercises, math word problems,
    #      narratives, multilingual — the de-facto standard for GGUF
    #      imatrix across bartowski's entire model library.
    #
    #   2. OpenThoughts-114k subset (thinking traces):
    #      Chain-of-thought reasoning with <think>…</think> blocks from
    #      open-thoughts/OpenThoughts-114k (public HuggingFace dataset).
    #      Calibrates the thinking layers that Qwen3 uses heavily.
    #
    #   3. glaive-function-calling-v2 subset (tool calling):
    #      JSON function-call turn pairs from glaiveai/glaive-function-
    #      calling-v2 (public HuggingFace dataset).  Covers the structured
    #      output patterns used by tool-call workflows.
    #
    # Mix: 60 % bartowski, 25 % thinking, 15 % tool-calling.
    # Each source is truncated so no single one dominates.
    # Falls back gracefully: if a HuggingFace fetch fails the remaining
    # sources still produce a valid corpus.
    # ------------------------------------------------------------------
    echo "[quantize] downloading calibration corpus (bartowski v5 + thinking + tool-calling)..."
    python3 - <<'PYEOF'
import urllib.request, os, random, sys, json, textwrap

tmpdir = os.environ["_CORPUS_TMPDIR"]
random.seed(42)
parts = {"base": [], "thinking": [], "toolcall": []}

# ── 1. bartowski calibration_datav5 ──────────────────────────────────────────
BART_URL = (
    "https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d"
    "/raw/ce111d8971a07caebd8234ef336b2102d6c5fb85/calibration_datav5.txt"
)
print("[quantize]   source 1/3: bartowski calibration_datav5 ...", flush=True)
try:
    path = os.path.join(tmpdir, "bart_v5.txt")
    urllib.request.urlretrieve(BART_URL, path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [l.strip() for l in fh if len(l.strip()) > 40]
    # Keep up to 2 400 lines (60 % target share)
    parts["base"] = lines[:2400]
    print(f"[quantize]     {len(parts['base'])} lines", flush=True)
except Exception as exc:
    print(f"[quantize]   WARNING: bartowski download failed: {exc}", flush=True)

# ── 2. OpenThoughts-114k — thinking traces ────────────────────────────────────
# Public dataset; fetch via HuggingFace datasets parquet endpoint.
# Each record has 'conversations': [{role, content}, …].  The assistant turn
# contains <think>…</think> blocks which are the key calibration signal.
THINK_URL = (
    "https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k"
    "/resolve/main/data/train-00000-of-00006.parquet"
)
print("[quantize]   source 2/3: OpenThoughts-114k (thinking traces) ...", flush=True)
try:
    import importlib
    if importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"):
        import pandas as pd
        path = os.path.join(tmpdir, "thinking.parquet")
        urllib.request.urlretrieve(THINK_URL, path)
        df = pd.read_parquet(path, columns=["conversations"])
        thinking_lines = []
        for convs in df["conversations"]:
            for turn in (convs if isinstance(convs, list) else []):
                role    = turn.get("role", "") if isinstance(turn, dict) else ""
                content = turn.get("content", "") if isinstance(turn, dict) else ""
                if role == "assistant" and "<think>" in content and len(content) > 200:
                    # Emit a condensed excerpt so one record doesn't flood the corpus
                    thinking_lines.append(content[:2000])
                    break
        random.shuffle(thinking_lines)
        # Keep up to ~1 000 lines (25 % target share)
        parts["thinking"] = thinking_lines[:1000]
        print(f"[quantize]     {len(parts['thinking'])} thinking traces", flush=True)
    else:
        # Fall back: install pandas+pyarrow transiently
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import pandas as pd
        path = os.path.join(tmpdir, "thinking.parquet")
        urllib.request.urlretrieve(THINK_URL, path)
        df = pd.read_parquet(path, columns=["conversations"])
        thinking_lines = []
        for convs in df["conversations"]:
            for turn in (convs if isinstance(convs, list) else []):
                role    = turn.get("role", "") if isinstance(turn, dict) else ""
                content = turn.get("content", "") if isinstance(turn, dict) else ""
                if role == "assistant" and "<think>" in content and len(content) > 200:
                    thinking_lines.append(content[:2000])
                    break
        random.shuffle(thinking_lines)
        parts["thinking"] = thinking_lines[:1000]
        print(f"[quantize]     {len(parts['thinking'])} thinking traces", flush=True)
except Exception as exc:
    print(f"[quantize]   WARNING: thinking corpus failed: {exc}", flush=True)

# ── 3. glaive-function-calling-v2 — tool-call turns ──────────────────────────
# Public dataset; JSONL format.  Each record has 'chat' field containing
# a conversation with function-call and result turns in chatml format.
GLAIVE_URL = (
    "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2"
    "/resolve/main/data/train-00000-of-00001-5b71c9e9688399b7.parquet"
)
print("[quantize]   source 3/3: glaive function-calling v2 (tool calls) ...", flush=True)
try:
    import importlib
    if not importlib.util.find_spec("pandas"):
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    import pandas as pd
    path = os.path.join(tmpdir, "glaive.parquet")
    urllib.request.urlretrieve(GLAIVE_URL, path)
    df = pd.read_parquet(path, columns=["chat"])
    tool_lines = []
    for chat in df["chat"]:
        if isinstance(chat, str) and len(chat) > 100:
            # Each chat is a full multi-turn conversation string; take a slice
            tool_lines.append(chat[:3000])
    random.shuffle(tool_lines)
    # Keep up to ~600 lines (15 % target share)
    parts["toolcall"] = tool_lines[:600]
    print(f"[quantize]     {len(parts['toolcall'])} tool-call conversations", flush=True)
except Exception as exc:
    print(f"[quantize]   WARNING: tool-call corpus failed: {exc}", flush=True)

# ── Merge, shuffle, write ─────────────────────────────────────────────────────
all_lines = parts["base"] + parts["thinking"] + parts["toolcall"]
if not all_lines:
    print("[quantize] ERROR: all corpus sources failed — cannot build imatrix", flush=True)
    sys.exit(1)

random.shuffle(all_lines)

out = os.path.join(tmpdir, "corpus.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(all_lines))

mb = os.path.getsize(out) / 1024 / 1024
print(
    f"[quantize] corpus assembled: {out} ({mb:.1f} MB, {len(all_lines)} lines) "
    f"[base={len(parts['base'])} thinking={len(parts['thinking'])} "
    f"toolcall={len(parts['toolcall'])}]",
    flush=True,
)
PYEOF

    mkdir -p "$(dirname "$IMATRIX")"

    # ------------------------------------------------------------------
    # Detect GPUs and build -ts (tensor-split) + -ngl arguments.
    #
    # nvidia-smi is queried for each GPU's free VRAM; the -ts ratio is
    # proportional so llama.cpp distributes layers to fit available memory.
    # NGL=999 means "offload everything" — llama.cpp caps it to the actual
    # layer count, so this is always safe.
    # Falls back: multi-GPU → partial-GPU (floor to total-VRAM fit) → CPU.
    # ------------------------------------------------------------------
    _build_gpu_args() {
        local ngl=$1
        # Collect free MiB for each GPU (one per line)
        local vrams
        vrams=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
                2>/dev/null | tr -d ' ')
        local ngpu
        ngpu=$(echo "$vrams" | grep -c '^[0-9]' || echo 0)

        if [[ "$ngpu" -lt 1 || "$ngl" -eq 0 ]]; then
            # No GPUs or explicitly CPU-only
            echo "-ngl 0"
            return
        fi

        if [[ "$ngpu" -eq 1 ]]; then
            echo "-ngl $ngl"
            return
        fi

        # Multi-GPU: build comma-separated VRAM ratios for -ts
        # e.g. two GPUs with 22000 and 18000 MiB free → "-ts 22000,18000"
        local ts
        ts=$(echo "$vrams" | tr '\n' ',' | sed 's/,$//')
        echo "-ngl $ngl -ts $ts"
    }

    _run_imatrix() {
        local ngl=$1
        local gpu_args
        gpu_args=$(_build_gpu_args "$ngl")
        # shellcheck disable=SC2086  # intentional word-splitting of gpu_args
        echo "[quantize] running llama-imatrix (chunks=$CHUNKS, gpu_args='$gpu_args')..."
        "$LLAMA_BIN_DIR/llama-imatrix" \
            -m "$BF16_GGUF" \
            -f "$_CORPUS_TMPDIR/corpus.txt" \
            -o "$IMATRIX" \
            --chunks "$CHUNKS" \
            $gpu_args
    }

    # Attempt 1: requested NGL (with full multi-GPU split)
    if ! _run_imatrix "$NGL"; then
        echo "[quantize] WARNING: llama-imatrix failed with ngl=$NGL (likely OOM)." >&2

        # Estimate how many layers fit across all GPUs (BF16 ≈ 1.6 GB/layer)
        _total_free_mib=$(nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk '{s+=$1} END{print s+0}')
        _partial_ngl=$(( _total_free_mib / 1638 ))   # 1638 MiB ≈ 1.6 GB

        if [[ "$_partial_ngl" -gt 0 ]]; then
            echo "[quantize] WARNING: retrying with -ngl $_partial_ngl" \
                 "(estimated fit across all GPUs)." >&2
            rm -f "$IMATRIX"
            # Attempt 2: partial offload across all GPUs
            if ! _run_imatrix "$_partial_ngl"; then
                echo "[quantize] WARNING: partial-GPU run also failed;" \
                     "falling back to CPU." >&2
                rm -f "$IMATRIX"
                _run_imatrix 0
            fi
        else
            echo "[quantize] WARNING: no GPU VRAM available; falling back to CPU." >&2
            rm -f "$IMATRIX"
            _run_imatrix 0
        fi
    fi
    echo "[quantize] imatrix done: $IMATRIX"
    # corpus tmpdir is removed by the EXIT trap above
    fi  # end: else (no bundled imatrix)
fi

echo "[quantize] $BF16_GGUF -> $OUT_GGUF (262K-Balanced recipe)"
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
echo "[quantize] blk.40 FFN experts (ffn_gate/up/down_exps) -> q8_0 for CUDA graph compat."
echo "[quantize] blk.40 attn/norm/router/nextn remain BF16 (graph-safe as-is)."
echo "[quantize] eh_proj (MTP output) tensor is requantized to q8_0 at *load* time"
echo "[quantize] by llama-server's --mtp-requantize-output-tensor flag (docker/.env)."
echo "[quantize] copy/symlink $SRC_DIR/mmproj-BF16.gguf next to it (or requantize"
echo "[quantize] it separately) so entrypoint.sh's ENABLE_VISION=1 can find it."
echo "[quantize] (mmproj is already BF16 from Unsloth - left untouched either way.)"

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
if [[ ! -f "${IMATRIX:-}" ]]; then
    echo "[quantize] ERROR: imatrix not found at '${IMATRIX:-}'; run compute-imatrix.sh first." >&2
    echo "[quantize] (set IMATRIX=/path/to/file or CORPUS_FILE=/path/to/corpus.txt)" >&2
    exit 1
fi

echo "[quantize] $BF16_GGUF -> $OUT_GGUF (262K-Balanced recipe)"
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
echo "[quantize] blk.40 FFN experts (ffn_gate/up/down_exps) -> q8_0 for CUDA graph compat."
echo "[quantize] blk.40 attn/norm/router/nextn remain BF16 (graph-safe as-is)."
echo "[quantize] eh_proj (MTP output) tensor is requantized to q8_0 at *load* time"
echo "[quantize] by llama-server's --mtp-requantize-output-tensor flag (docker/.env)."
echo "[quantize] copy/symlink $SRC_DIR/mmproj-BF16.gguf next to it (or requantize"
echo "[quantize] it separately) so entrypoint.sh's ENABLE_VISION=1 can find it."
echo "[quantize] (mmproj is already BF16 from Unsloth - left untouched either way.)"
