#!/bin/bash
set -e

# Model configuration lives in config/models.ini (host) →
# /etc/llama-server/models.ini (container, mounted read-only).
# Add a [section] per model; [*] sets global defaults.
# Router supports single-model setups too — just define one section.

HOST=${LLAMA_HOST:-0.0.0.0}
PORT=${LLAMA_PORT:-8080}
API_KEY=${LLAMA_API_KEY:-}
UI_CONFIG_FILE=/etc/llama-server/webui-config.json
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

WEBUI_PATH=/etc/llama-server/webui

EXTRA_ARGS=()
[ "$MODELS_AUTOLOAD" != "on" ] && EXTRA_ARGS+=(--no-models-autoload)
[ -n "$API_KEY" ]               && EXTRA_ARGS+=(--api-key "$API_KEY")
# --path overrides the embedded UI \u2014 useful for local development without a rebuild.
[ -d "$WEBUI_PATH" ]            && EXTRA_ARGS+=(--path "$WEBUI_PATH")

exec llama-server \
    --host "$HOST" \
    --port "$PORT" \
    --models-preset "$MODELS_PRESET" \
    --models-max "$MODELS_MAX" \
    --ui-config-file "$UI_CONFIG_FILE" \
    --ui-mcp-proxy \
    "${EXTRA_ARGS[@]}" \
    "$@"
