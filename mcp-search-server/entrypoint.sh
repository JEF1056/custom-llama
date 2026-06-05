#!/bin/bash
set -e

# Verify Playwright browsers exist
if [ ! -d "${PLAYWRIGHT_BROWSERS_PATH:-/opt/playwright}" ]; then
    echo "ERROR: Playwright browsers not found at $PLAYWRIGHT_BROWSERS_PATH"
    exit 1
fi

# Create mcp-files directories and clean stale files (>24h)
mkdir -p /app/mcp-files/screenshots
find /app/mcp-files -type f -mtime +1 -delete 2>/dev/null || true

echo "=== MCP Search Server ==="
echo "  Host: ${MCP_SERVER_HOST:-0.0.0.0}:${MCP_SERVER_PORT:-3100}"
echo "  Search: ${SEARCH_ENGINE:-duckduckgo} (max ${MAX_RESULTS:-10})"
echo "  Cache: ${CACHE_ENABLED:-true} (TTL ${CACHE_TTL:-3600}s)"

exec python -m src.server
