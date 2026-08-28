FROM ghcr.io/thushan/olla:latest

# Mount config file
COPY config.yaml /app/config/config.local.yaml

EXPOSE 40114

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -q -O - http://localhost:40114/internal/health || exit 1
