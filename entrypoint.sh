#!/bin/bash
set -e

# Model configuration lives in config/models.ini (host) →
# /etc/llama-server/models.ini (container, mounted read-only).
# Add a [section] per model; [*] sets global defaults.
# Router supports single-model setups too — just define one section.

HOST=${LLAMA_HOST:-0.0.0.0}
PORT=${LLAMA_PORT:-8080}
API_KEY=${LLAMA_API_KEY:-}
WEBUI_CONFIG_FILE=/etc/llama-server/webui-config.json
MODELS_PRESET=/etc/llama-server/models.ini
MODELS_MAX=${LLAMA_MODELS_MAX:-1}
MODELS_AUTOLOAD=${LLAMA_MODELS_AUTOLOAD:-on}

if [ ! -f "$MODELS_PRESET" ]; then
    echo "ERROR: models preset not found: $MODELS_PRESET"
    echo "Mount config/models.ini to /etc/llama-server/models.ini"
    exit 1
fi

# Create slot-save-path directories declared in the preset
grep -E '^\s*slot-save-path\s*=' "$MODELS_PRESET" | sed 's/.*=\s*//' | while IFS= read -r slot_dir; do
    mkdir -p "$slot_dir"
done

echo "Starting llama-server (router mode)"
echo "  Host:              $HOST:$PORT"
echo "  Preset:            $MODELS_PRESET"
echo "  Max loaded models: $MODELS_MAX"
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
