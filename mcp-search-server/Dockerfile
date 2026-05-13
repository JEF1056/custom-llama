# Stage 1: Build stage
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies (rarely changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (changes when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy pyproject.toml and install the application (changes when pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Copy source code (changes frequently — placed last for optimal caching)
COPY src/ ./src/

# Stage 2: Runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install system dependencies for Playwright (chromium) (rarely changes)
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

# Copy installed Python packages from builder stage (changes when Python dependencies change)
COPY --from=builder /install /usr/local

# Copy pyproject.toml (changes rarely)
COPY --from=builder /build/pyproject.toml .

# Copy entrypoint script and make it executable (changes rarely)
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Install Playwright browsers as root (very expensive, rarely changes — placed before source code)
# Use --with-deps to install system dependencies and browsers in one step
RUN playwright install --with-deps chromium && \
    # Verify the installation succeeded
    ls -la /root/.cache/ms-playwright/

# Create non-root user (rarely changes)
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

# Copy Playwright browsers to appuser's cache directory so they're accessible
# Create the directory structure and copy all browser binaries
RUN mkdir -p /home/appuser/.cache/ms-playwright && \
    cp -r /root/.cache/ms-playwright/* /home/appuser/.cache/ms-playwright/ 2>/dev/null || true && \
    chown -R appuser:appuser /home/appuser/.cache/ms-playwright && \
    # Verify the copy succeeded
    ls -la /home/appuser/.cache/ms-playwright/

# Switch to non-root user (rarely changes)
USER appuser

# Set HOME to appuser's home directory so Playwright finds browsers in the right place
ENV HOME=/home/appuser

# Copy source code (changes frequently — placed last for optimal caching)
COPY --from=builder /build/src/ ./src/

# Expose the MCP server port (never changes)
EXPOSE ${MCP_SERVER_PORT:-3100}

# Health check - MCP server uses SSE transport on HTTP
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${MCP_SERVER_PORT:-3100}/health || exit 1

# Set entrypoint (rarely changes)
ENTRYPOINT ["./entrypoint.sh"]
