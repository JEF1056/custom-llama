#!/bin/bash
set -e

echo "=== MCP Search Server ==="
echo "Starting up..."

# Wait for Playwright browsers to be ready
echo "Checking Playwright browser installation..."
if [ ! -d "/root/.cache/ms-playwright" ]; then
    echo "Installing Playwright browsers..."
    playwright install chromium
fi

# Verify Playwright is working
echo "Verifying Playwright installation..."
if playwright install --dry-run 2>/dev/null; then
    echo "Playwright browsers are ready!"
else
    echo "Warning: Playwright browsers may not be fully installed"
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Host: ${MCP_SERVER_HOST:-0.0.0.0}"
echo "  Port: ${MCP_SERVER_PORT:-3100}"
echo "  Search Engine: ${SEARCH_ENGINE:-duckduckgo}"
echo "  Max Results: ${MAX_RESULTS:-10}"
echo "  Cache Enabled: ${CACHE_ENABLED:-true}"
echo "  Cache TTL: ${CACHE_TTL:-3600}s"
echo ""

# Start the MCP server
echo "Starting MCP server..."
exec python -m src.server
