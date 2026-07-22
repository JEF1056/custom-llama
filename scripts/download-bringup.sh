#!/usr/bin/env bash
#
# Bring-up helper: fetch ONE public pre-quantized GGUF (+ mmproj) straight from
# Hugging Face via curl, for smoke-testing the engine (hybrid arch, Hadamard
# KV, MTP, vision) before the in-house "262K-Balanced" recipe (scripts/
# quantize.sh) is ready. Writes into the same /models volume the server reads
# from, using the plain filenames MODEL_SOURCE=local expects - so once this
# finishes, just set GGUF_FILE/MMPROJ_FILE in docker/.env to what was
# downloaded here and start the server normally.
#
# Meant to run as the `model-bringup` compose service/profile (see
# docker-compose.yml), so re-running it (e.g. after a --force-recreate) is
# idempotent: curl -C - resumes/no-ops on an already-complete file.
set -euo pipefail

HF_REPO=${HF_REPO:-llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF}
BRINGUP_FILE=${BRINGUP_FILE:-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf}
MMPROJ_SRC_FILE=${MMPROJ_SRC_FILE:-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf}
DEST_DIR=${DEST_DIR:-/models}
GGUF_FILE=${GGUF_FILE:-qwen36-bringup.gguf}
MMPROJ_FILE=${MMPROJ_FILE:-mmproj-BF16.gguf}

mkdir -p "$DEST_DIR"

echo "[bringup] $HF_REPO / $BRINGUP_FILE -> $DEST_DIR/$GGUF_FILE"
curl -fL --retry 5 -C - -o "$DEST_DIR/$GGUF_FILE" \
    "https://huggingface.co/${HF_REPO}/resolve/main/${BRINGUP_FILE}"

echo "[bringup] $HF_REPO / $MMPROJ_SRC_FILE -> $DEST_DIR/$MMPROJ_FILE"
curl -fL --retry 5 -C - -o "$DEST_DIR/$MMPROJ_FILE" \
    "https://huggingface.co/${HF_REPO}/resolve/main/${MMPROJ_SRC_FILE}"

echo "[bringup] done. Set docker/.env: MODEL_SOURCE=local GGUF_FILE=$GGUF_FILE MMPROJ_FILE=$MMPROJ_FILE"
