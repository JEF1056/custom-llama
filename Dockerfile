# =============================================================================
# SGLang server — JEF1056/sglang-turboquant + TurboQuant PR #23135
#
# Pre-requisite: merge sgl-project/sglang PR #23135 into JEF1056/sglang-turboquant
# on GitHub BEFORE running `docker compose build sglang-server`.
#
# PR #23135: fused Triton 4-bit KV cache compression (3.88x compression,
# 93–105% bf16 decode throughput, CUDA graph compatible).
#
# Model preparation (download / quantize / convert) is handled by the separate
# Dockerfile.convert image. This image only serves /models via SGLang.
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 ships Python 3.10, which meets SGLang's requirements.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip \
      git curl cmake build-essential ninja-build \
      libssl-dev pkg-config libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 12.4) first — sgl-kernel is a torch CUDA extension
# and requires torch to be present before its wheel can be built.
RUN python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# Clone fork — PR #23135 must be merged into main before this step.
RUN git clone https://github.com/JEF1056/sglang-turboquant.git /sglang

# Build sgl-kernel: CUDA-compiled C++ extensions + Triton kernels.
# This is the step that takes the most time (~10–20 min on first build).
WORKDIR /sglang/sgl-kernel
RUN python3 -m pip install --no-cache-dir .

# Install SGLang with all extras (flashinfer, triton, vllm attention backends).
# 'all' includes every optional attention and sampling backend.
WORKDIR /sglang/python
RUN python3 -m pip install --no-cache-dir ".[all]"

ENV MODEL_DIR=/models
RUN mkdir -p /models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && sed -i 's/\r$//' /entrypoint.sh && \
    python3 -c "import sglang; print('SGLang', sglang.__version__, 'OK')"

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=300s \
    CMD curl -f http://localhost:${SGLANG_PORT:-8080}/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
