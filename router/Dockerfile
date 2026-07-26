FROM python:3.12-slim

# Install vllm-sr
RUN pip install --no-cache-dir vllm-sr==0.3.0

# Create app directory
WORKDIR /app

# Copy configuration
COPY config.yaml /app/config.yaml

# Expose ports
# 8888 - API (OpenAI-compatible)
# 8700 - Dashboard
# 9190 - Prometheus metrics
EXPOSE 8888 8700 9190

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8888/v1/models')" || exit 1

# Start vllm-sr with the config
# HF_TOKEN is required for downloading embedding models
CMD ["vllm-sr", "serve", "--config", "/app/config.yaml"]
