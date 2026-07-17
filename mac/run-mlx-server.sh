#!/usr/bin/env bash
#
# Launches the Bonsai-27B MLX server. Invoked by the LaunchAgent.
#
# DSpark speculative decoding is OFF here: it does not speed up batch-1 decode
# on Apple Silicon today. Native tool calling and the full 262K context are
# available by default (see notes below).
#
# Vision: image input on Apple Silicon is served through mlx-vlm, which only
# supports the TERNARY (2-bit) 27B MLX build; the 1-bit MLX build is text-only
# today (it needs the PrismML MLX fork, which has no VLM path yet). Set
# ENABLE_VISION=1 (at install time too) to serve the ternary 27B with images.
set -euo pipefail

BONSAI_HOME=${BONSAI_HOME:-$HOME/.bonsai}
DEMO_DIR="$BONSAI_HOME/Bonsai-demo"

export BONSAI_MODEL=${BONSAI_MODEL:-27B}
export BONSAI_SPECULATIVE=0

# Vision routing (see header). ENABLE_VISION=1 selects the ternary family and
# lets the demo's start_mlx_server.sh route to mlx-vlm (image input); otherwise
# stay on the 1-bit text-only mlx_lm server. install.sh must have run with the
# same ENABLE_VISION so the matching model + mlx-vlm venv exist.
case "${ENABLE_VISION:-0}" in
    1|true|yes|on) VISION_ON=1 ;;
    *)             VISION_ON=0 ;;
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
# Max context: mlx-lm keeps an unbounded, dynamically growing KV cache unless
# --max-kv-size is passed, so we deliberately do NOT pass it. The full 262K
# window is available, limited only by available unified memory. (BONSAI_KV4 is
# a llama.cpp-only knob and has no effect on MLX.)

MLX_HOST=${MLX_HOST:-0.0.0.0}
MLX_PORT=${MLX_PORT:-8081}

cd "$DEMO_DIR"

# Activate the venv created by setup.sh (mlx-lm lives here).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# start_mlx_server.sh forwards extra args to the MLX server.
exec ./scripts/start_mlx_server.sh \
    --host "$MLX_HOST" \
    --port "$MLX_PORT" \
    ${EXTRA_ARGS:-}
