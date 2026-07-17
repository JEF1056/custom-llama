#!/usr/bin/env bash
#
# Launches the Bonsai-27B (1-bit) MLX server. Invoked by the LaunchAgent.
#
# DSpark speculative decoding is OFF here: it does not speed up batch-1 decode
# on Apple Silicon today. Native tool calling and the full 262K context are
# available by default (see notes below).
set -euo pipefail

BONSAI_HOME=${BONSAI_HOME:-$HOME/.bonsai}
DEMO_DIR="$BONSAI_HOME/Bonsai-demo"

export BONSAI_FAMILY=bonsai
export BONSAI_MODEL=27B
export BONSAI_SPECULATIVE=0

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
