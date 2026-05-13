# Stage 1: Build stage
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the application
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install system dependencies for Playwright (chromium)
# Based on https://github.com/microsoft/playwright/blob/main/packages/playwright-core/src/server/linux/linuxBrowserRunner.js
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Core dependencies
    libglib2.0-0 \
    libglib2.0-0t64 \
    libglib2.0-bin \
    libglib2.0-data \
    libglib2.0-dev \
    # Additional dependencies for browser automation
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libglib2.0-0t64 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxss1 \
    libxtst6 \
    fonts-liberation \
    fonts-dejavu-core \
    # Other utilities
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY --from=builder /build/src/ ./src/
COPY --from=builder /build/pyproject.toml .

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Install Playwright browsers
RUN playwright install --with-deps chromium

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose the MCP server port
EXPOSE ${MCP_SERVER_PORT:-3100}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${MCP_SERVER_PORT:-3100}/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
