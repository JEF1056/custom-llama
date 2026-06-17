# Stage 1: Build stage
FROM python:3.12-slim AS builder

WORKDIR /build

# uv: Rust-based pip replacement — 10-100x faster dependency resolution & install.
COPY --link --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install build dependencies (rarely changes)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc g++

# Install all Python dependencies in one layer
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --prefix=/install -r requirements.txt .

# Stage 2: Runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed Python packages from builder
COPY --link --from=builder /install /usr/local

# Install the real Google Chrome browser + all its system deps in one shot.
# patchright drives genuine Google Chrome (channel="chrome"), which is far
# harder to fingerprint than bundled Chromium / Chrome-for-Testing. xvfb lets
# Chrome run headful (the strongest anti-detection posture) on a headless host.
# Browsers go to /opt/playwright (owned root:root, world-readable).
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates xvfb xauth && \
    patchright install --with-deps chrome && \
    patchright install --with-deps chromium && \
    chmod -R o+rX /opt/playwright

# Create non-root user and pre-create all writable dirs with correct ownership.
# mcp-files/screenshots  — written by entrypoint and tools at runtime
# .cache/python          — mounted as a named volume (Docker sets root:root on
#                          first mount, so we pre-create and chown here so the
#                          volume inherits the correct owner on first use)
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /home/appuser -s /sbin/nologin -m appuser && \
    mkdir -p /app/mcp-files/screenshots /app/.cache/python && \
    chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser

# Copy entrypoint and source (changes frequently — placed last)
COPY --chown=appuser:appuser entrypoint.sh .
RUN chmod +x entrypoint.sh
COPY --chown=appuser:appuser --from=builder /build/src/ ./src/

EXPOSE ${MCP_SERVER_PORT:-3100}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${MCP_SERVER_PORT:-3100}/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
