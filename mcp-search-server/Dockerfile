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

# Shared browser path — avoids root→appuser copy dance
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# Copy installed Python packages from builder
COPY --link --from=builder /install /usr/local

# Install Playwright browser + all its system deps in one shot.
# --with-deps handles the exact apt packages Chromium needs — no manual list.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    playwright install --with-deps chromium

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin -m appuser && \
    chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser

# Copy entrypoint and source (changes frequently — placed last)
COPY --chown=appuser:appuser entrypoint.sh .
COPY --link --from=builder /build/src/ ./src/

EXPOSE ${MCP_SERVER_PORT:-3100}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${MCP_SERVER_PORT:-3100}/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
