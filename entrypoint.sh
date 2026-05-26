#!/bin/bash
set -e

# =============================================================================
# llama-server entrypoint
#
# Router mode (default): all model configuration lives in LLAMA_MODELS_PRESET
# (scripts/models.ini → /etc/llama-server/models.ini inside the container).
# Each model section defines its own ctx-size, KV config, spec params, etc.
# Global defaults shared across all models are in the [*] section.
#
# Single-model fallback: set LLAMA_ROUTER=off and provide LLAMA_MODEL or
# MODEL_NAME + QUANT. Model-level flags must then be passed via extra args
# or environment variables recognised by llama-server.
# =============================================================================

HOST=${LLAMA_HOST:-0.0.0.0}
PORT=${LLAMA_PORT:-8080}
API_KEY=${LLAMA_API_KEY:-}
WEBUI_CONFIG_FILE=${LLAMA_WEBUI_CONFIG_FILE:-/etc/llama-server/webui-config.json}

# Router mode settings
ROUTER=${LLAMA_ROUTER:-on}
MODELS_PRESET=${LLAMA_MODELS_PRESET:-/etc/llama-server/models.ini}
MODELS_MAX=${LLAMA_MODELS_MAX:-1}
MODELS_AUTOLOAD=${LLAMA_MODELS_AUTOLOAD:-on}

# =============================================================================
# Router mode
# =============================================================================
if [ "$ROUTER" = "on" ]; then
    if [ ! -f "$MODELS_PRESET" ]; then
        echo "ERROR: models preset not found: $MODELS_PRESET"
        echo "Set LLAMA_MODELS_PRESET or place models.ini at /etc/llama-server/models.ini"
        exit 1
    fi

    echo "Starting llama-server (router mode)"
    echo "  Host:              $HOST:$PORT"
    echo "  Preset:            $MODELS_PRESET"
    echo "  Max loaded models: $MODELS_MAX"
    echo "  Autoload:          $MODELS_AUTOLOAD"
    [ -n "$API_KEY" ] && echo "  API key:           (set)"

    exec llama-server \
        --host "$HOST" \
        --port "$PORT" \
        --models-preset "$MODELS_PRESET" \
        --models-max "$MODELS_MAX" \
        $([ "$MODELS_AUTOLOAD" != "on" ] && echo "--no-models-autoload") \
        ${API_KEY:+--api-key "$API_KEY"} \
        ${WEBUI_CONFIG_FILE:+--webui-config-file "$WEBUI_CONFIG_FILE"} \
        "$@"
fi

# =============================================================================
# Single-model fallback (LLAMA_ROUTER=off)
# Model-level params (ctx, KV cache, sampling, etc.) must be provided via
# extra CLI args passed to this script or via llama-server env vars.
# =============================================================================

# Resolve model path
if [ -n "$LLAMA_MODEL" ]; then
    MODEL="$LLAMA_MODEL"
elif [ -n "$MODEL_NAME" ]; then
    QUANT="${QUANT:-Q4_K_M}"
    MODEL="/models/${MODEL_NAME}-${QUANT}.gguf"
else
    MODEL=/models/model.gguf
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    echo ""
    echo "Prepare the model first using the convert image, for example:"
    if [ -n "$MODEL_NAME" ]; then
        echo "  docker compose run --rm llama-convert download $MODEL_NAME --quant ${QUANT:-Q4_K_M}"
        echo "  # or for safetensors-only models:"
        echo "  docker compose run --rm llama-convert convert-st $MODEL_NAME --quant TQ2_0"
    else
        echo "  docker compose run --rm llama-convert download <model-name> --quant <quant>"
    fi
    exit 1
fi

MMPROJ=${LLAMA_MMPROJ:-}
if [ -z "$MMPROJ" ] && [ -n "$MODEL_NAME" ]; then
    _auto="/models/${MODEL_NAME}-mmproj.gguf"
    [ -f "$_auto" ] && MMPROJ="$_auto"
    unset _auto
fi
if [ -n "$MMPROJ" ] && [ ! -f "$MMPROJ" ]; then
    echo "WARNING: mmproj not found: $MMPROJ — disabling multimodal"
    MMPROJ=""
fi

echo "Starting llama-server (single-model mode)"
echo "  Host:  $HOST:$PORT"
echo "  Model: $MODEL"
[ -n "$MMPROJ" ] && echo "  Mmproj: $MMPROJ"
[ -n "$API_KEY" ] && echo "  API key: (set)"

MMFLAGS=""
[ -n "$MMPROJ" ] && MMFLAGS="--mmproj $MMPROJ"

exec llama-server \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    ${API_KEY:+--api-key "$API_KEY"} \
    ${WEBUI_CONFIG_FILE:+--webui-config-file "$WEBUI_CONFIG_FILE"} \
    $MMFLAGS \
    --jinja \
    "$@"
