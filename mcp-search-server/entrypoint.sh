#!/bin/bash
set -e

# The container starts as root so we can make the bind-mounted output dir
# writable regardless of its host ownership/uid, then drop to the unprivileged
# appuser and re-exec this same script. (A bind mount keeps the host's
# ownership, which usually won't match appuser's uid — the cause of
# "Permission denied" when saving screenshots.)
if [ "$(id -u)" = "0" ]; then
    mkdir -p /app/mcp-files/screenshots 2>/dev/null || true
    chown -R appuser:appuser /app/mcp-files 2>/dev/null || true
    # Xvfb needs /tmp/.X11-unix to exist with sticky bit before it drops to
    # a non-root uid; otherwise it gets EACCES and Chrome finds no display.
    mkdir -p /tmp/.X11-unix 2>/dev/null || true
    chmod 1777 /tmp/.X11-unix 2>/dev/null || true
    exec gosu appuser "$0" "$@"
fi

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

# Run Chrome headful under a virtual display when available — a real (non-
# headless) browser is substantially harder to fingerprint as a bot. We start
# Xvfb ourselves and exec the server directly rather than using `xvfb-run`,
# which deadlocks as PID 1 when its child's stdout is the container log pipe.
# Falls back to headless if Xvfb is missing (e.g. local dev) so the server
# still starts.
if command -v Xvfb >/dev/null 2>&1; then
    export BROWSER_HEADLESS="${BROWSER_HEADLESS:-false}"
    export DISPLAY="${DISPLAY:-:99}"
    echo "  Browser: headful via Xvfb on ${DISPLAY} (BROWSER_HEADLESS=${BROWSER_HEADLESS})"
    # -ac disables X access control so Chrome can connect without xauth.
    Xvfb "${DISPLAY}" -screen 0 1920x1080x24 -ac +extension GLX +render -noreset -nolisten tcp &
    # Wait briefly for the X socket so Chrome's first launch finds the display.
    sock="/tmp/.X11-unix/X${DISPLAY#:}"
    for _ in $(seq 1 50); do [ -e "$sock" ] && break; sleep 0.1; done
    exec python -m src.server
else
    export BROWSER_HEADLESS="${BROWSER_HEADLESS:-true}"
    echo "  Browser: headless (Xvfb unavailable, BROWSER_HEADLESS=${BROWSER_HEADLESS})"
    exec python -m src.server
fi
