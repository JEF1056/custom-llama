#!/bin/bash
set -e

echo "=== MCP Search Server ==="
echo "Starting up..."

# Set HOME to appuser's home directory so Playwright finds browsers in the right place
export HOME=/home/appuser

# Wait for Playwright browsers to be ready
echo "Checking Playwright browser installation..."
# Check both root and appuser cache directories
if [ ! -d "/root/.cache/ms-playwright" ] && [ ! -d "/home/appuser/.cache/ms-playwright" ]; then
    echo "Installing Playwright browsers as root..."
    # Run as root to install browsers (need root for --with-deps)
    su -c "HOME=/root playwright install --with-deps chromium" root
    # Copy to appuser's cache directory
    mkdir -p /home/appuser/.cache/ms-playwright
    cp -r /root/.cache/ms-playwright/* /home/appuser/.cache/ms-playwright/ 2>/dev/null || true
    chown -R appuser:appuser /home/appuser/.cache/ms-playwright
    echo "Browsers installed and copied to appuser cache"
else
    # Browsers already exist, ensure they're accessible to appuser
    if [ -d "/root/.cache/ms-playwright" ] && [ ! -d "/home/appuser/.cache/ms-playwright" ]; then
        echo "Copying browsers from root cache to appuser cache..."
        mkdir -p /home/appuser/.cache/ms-playwright
        cp -r /root/.cache/ms-playwright/* /home/appuser/.cache/ms-playwright/ 2>/dev/null || true
        chown -R appuser:appuser /home/appuser/.cache/ms-playwright
    fi
fi

# Verify Playwright browsers exist
echo "Verifying Playwright installation..."
if [ -d "/home/appuser/.cache/ms-playwright" ]; then
    echo "Playwright browsers found:"
    ls -la /home/appuser/.cache/ms-playwright/
    echo "Playwright browsers are ready!"
else
    echo "ERROR: Playwright browsers not found! Search will not work."
    exit 1
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
