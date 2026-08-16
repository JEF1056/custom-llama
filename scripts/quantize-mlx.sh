#!/usr/bin/env bash
#
# Convert and quantize Qwen3.8-27B-heretic-ara into pure 4-bit MXFP4 for MLX.
# Uses mlx_vlm.convert with --quantize --q-group-size 32 --q-mode mxfp4
#
# Usage:
#   bash quantize-mlx.sh [--hf-path <repo-or-dir>] [--output <quantized-dir>]
#
set -euo pipefail

HF_PATH=${HF_PATH:-"trohrbaugh/Qwen3.8-27B-heretic-ara"}
OUTPUT_DIR=${OUTPUT_DIR:-"$HOME/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4"}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-path|--input) HF_PATH="$2"; shift 2 ;;
        --output)          OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "[quantize-mlx] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "[quantize-mlx] Source: $HF_PATH"
echo "[quantize-mlx] Output: $OUTPUT_DIR"
echo "[quantize-mlx] Mode:   Pure 4-bit MXFP4 (group_size=32)"

mkdir -p "$(dirname "$OUTPUT_DIR")"

python3 -m mlx_vlm.convert \
    --hf-path "$HF_PATH" \
    --mlx-path "$OUTPUT_DIR" \
    -q \
    --q-group-size 32 \
    --q-mode mxfp4

echo "[quantize-mlx] Done: $OUTPUT_DIR"
