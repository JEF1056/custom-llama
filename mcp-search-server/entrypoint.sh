#!/bin/bash
set -e

# Verify Playwright browsers exist
if [ ! -d "${PLAYWRIGHT_BROWSERS_PATH:-/opt/playwright}" ]; then
    echo "ERROR: Playwright browsers not found at $PLAYWRIGHT_BROWSERS_PATH"
    exit 1
fi

# Ensure mcp-files dirs exist (pre-created in image; this is a safety net)
# and clean stale files (>24h)
mkdir -p /app/mcp-files/screenshots 2>/dev/null || true
find /app/mcp-files -type f -mtime +1 -delete 2>/dev/null || true

echo "=== MCP Search Server ==="
echo "  Host: ${MCP_SERVER_HOST:-0.0.0.0}:${MCP_SERVER_PORT:-3100}"
echo "  Search: ${SEARCH_ENGINE:-duckduckgo} (max ${MAX_RESULTS:-10})"
echo "  Cache: ${CACHE_ENABLED:-true} (TTL ${CACHE_TTL:-3600}s)"

# Run Chromium headful under a virtual display when available — a real (non-
# headless) browser is substantially harder to fingerprint as a bot. Falls back
# to headless if Xvfb is missing (e.g. local dev) so the server still starts.
if command -v xvfb-run >/dev/null 2>&1; then
    export BROWSER_HEADLESS="${BROWSER_HEADLESS:-false}"
    echo "  Browser: headful via Xvfb (BROWSER_HEADLESS=${BROWSER_HEADLESS})"
    exec xvfb-run -a --server-args="-screen 0 1920x1080x24 -ac +extension GLX +render -noreset" \
        python -m src.server
else
    export BROWSER_HEADLESS="${BROWSER_HEADLESS:-true}"
    echo "  Browser: headless (Xvfb unavailable, BROWSER_HEADLESS=${BROWSER_HEADLESS})"
    exec python -m src.server
fi
