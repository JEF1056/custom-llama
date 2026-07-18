#!/usr/bin/env bash
#
# Phase 2 step 3: compute our own diverse imatrix (chat/code/reasoning/
# tool-calling; deliberately NOT Wikipedia, per the APEX-aligned rationale in
# docs/iqllama-migration-plan.md section 4c) from the merged BF16 GGUF.
#
# Compare its downstream quality against Unsloth's shipped
# imatrix_unsloth.gguf_file in the Phase 7e perplexity check; keep whichever
# scores better.
set -euo pipefail

SRC_DIR=${SRC_DIR:-/models/qwen36-src}
BF16_GGUF=${BF16_GGUF:-$SRC_DIR/qwen36-bf16.gguf}
CORPUS_FILE=${CORPUS_FILE:-/models/qwen36-src/diverse_corpus.txt}
OUT_IMATRIX=${OUT_IMATRIX:-/models/qwen36-src/qwen36.imatrix}
CHUNKS=${CHUNKS:-200}
NGL=${NGL:-999}
LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-/opt/iqllama/bin}

if [[ ! -f "$BF16_GGUF" ]]; then
    echo "[imatrix] ERROR: $BF16_GGUF not found; run download-source-gguf.sh first." >&2
    exit 1
fi
if [[ ! -f "$CORPUS_FILE" ]]; then
    echo "[imatrix] ERROR: $CORPUS_FILE not found." >&2
    echo "[imatrix] Provide a diverse chat/code/reasoning/tool-calling corpus" >&2
    echo "[imatrix] (no Wikipedia - see migration plan section 4c) at that path." >&2
    exit 1
fi

echo "[imatrix] computing imatrix from $BF16_GGUF over $CORPUS_FILE ($CHUNKS chunks)"
"$LLAMA_BIN_DIR/llama-imatrix" \
    -m "$BF16_GGUF" \
    -f "$CORPUS_FILE" \
    -o "$OUT_IMATRIX" \
    --chunks "$CHUNKS" \
    -ngl "$NGL"

echo "[imatrix] done: $OUT_IMATRIX"
