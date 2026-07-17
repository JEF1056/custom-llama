#!/usr/bin/env bash
#
# Launches the Bonsai-27B MLX server. Invoked by the LaunchAgent.
#
# DSpark speculative decoding is OFF here: it does not speed up batch-1 decode
# on Apple Silicon today. Native tool calling and the full 262K context are
# available by default (see notes below).
#
# Prompt caching: Automatic Prefix Caching (APC) is enabled (APC_ENABLED=1) so
# mlx-vlm/mlx-lm reuse a shared prefix's KV across requests; on the vision path
# the block hash folds in an image content hash so cached text prefixes stay
# correct across different images. KV cache is 4-bit quantized (--kv-bits 4),
# the MLX analogue of llama.cpp's q4_0.
#
# Vision: enabled by default. Image input on Apple Silicon is served through
# mlx-vlm, which only supports the TERNARY (2-bit) 27B MLX build; the 1-bit MLX
# build is text-only today (it needs the PrismML MLX fork, which has no VLM path
# yet). Set ENABLE_VISION=0 for the leaner 1-bit text-only server.
set -euo pipefail

BONSAI_HOME=${BONSAI_HOME:-$HOME/.bonsai}
DEMO_DIR="$BONSAI_HOME/Bonsai-demo"

export BONSAI_MODEL=${BONSAI_MODEL:-27B}
export BONSAI_SPECULATIVE=0

# Prompt caching: enable Automatic Prefix Caching in mlx-vlm/mlx-lm (the servers
# leave it off by default). apc.from_env() reads this straight from the process
# environment, so exporting it is all that's needed. Set APC_ENABLED=0 to disable.
export APC_ENABLED=${APC_ENABLED:-1}

# KV cache quantization: 4-bit uniform, the MLX analogue of llama.cpp q4_0.
# Set MLX_KV_BITS= (empty) to keep a full-precision KV cache.
MLX_KV_BITS=${MLX_KV_BITS:-4}

# Vision routing (see header). ENABLE_VISION=1 selects the ternary family and
# lets the demo's start_mlx_server.sh route to mlx-vlm (image input); otherwise
# stay on the 1-bit text-only mlx_lm server. install.sh must have run with the
# same ENABLE_VISION so the matching model + mlx-vlm venv exist.
case "${ENABLE_VISION:-1}" in
    0|false|no|off) VISION_ON=0 ;;
    *)              VISION_ON=1 ;;
esac
if [[ "$VISION_ON" == "1" ]]; then
    export BONSAI_FAMILY=${BONSAI_FAMILY:-ternary}
    export BONSAI_MLX_VLM=1
    if [[ "$BONSAI_FAMILY" != "ternary" ]]; then
        echo "[bonsai-mlx] WARNING: ENABLE_VISION=1 but BONSAI_FAMILY=$BONSAI_FAMILY;" \
             "MLX image input only works on the ternary build, so vision stays OFF." >&2
    fi
else
    export BONSAI_FAMILY=${BONSAI_FAMILY:-bonsai}
    export BONSAI_MLX_VLM=0
fi

# Tool calling: the MLX server emits native OpenAI-style tool_calls for the 27B
# out of the box (no flag needed).
#
# Max context: the model's native max is 262144 (~262K). mlx-lm/mlx-vlm keep an
# unbounded, dynamically growing KV cache unless --max-kv-size is passed, so we
# deliberately do NOT cap it - the full native window is available, limited only
# by unified memory. KV entries are 4-bit quantized (--kv-bits) to stretch it.

MLX_HOST=${MLX_HOST:-0.0.0.0}
MLX_PORT=${MLX_PORT:-8081}

cd "$DEMO_DIR"

# Activate the venv created by setup.sh (mlx-lm lives here).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Pass the 4-bit KV flag through to whichever MLX server start_mlx_server.sh
# picks (mlx-vlm for vision, mlx-lm for text - both accept --kv-bits). Leave
# MLX_KV_BITS empty to skip it.
KV_BITS_ARG=""
if [[ -n "$MLX_KV_BITS" ]]; then
    KV_BITS_ARG="--kv-bits $MLX_KV_BITS"
fi

# start_mlx_server.sh forwards extra args to the MLX server.
exec ./scripts/start_mlx_server.sh \
    --host "$MLX_HOST" \
    --port "$MLX_PORT" \
    ${KV_BITS_ARG} \
    ${EXTRA_ARGS:-}
