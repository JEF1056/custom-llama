# =============================================================================
# Stage 1: Builder - Build llama.cpp with CUDA for llama-server
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  cmake \
  build-essential \
  wget \
  curl \
  python3 \
  python3-pip \
  pkg-config \
  libopenblas-dev && \
  rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recursive \
  --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git /llama.cpp

WORKDIR /llama.cpp

RUN cmake \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_HBM=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  . && \
  cmake --build . --parallel $(nproc)

RUN rm -rf /llama.cpp/.git


# =============================================================================
# Stage 2: Build CPU-only static llama-quantize
# =============================================================================
FROM python:3.11-slim AS model-manager

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  cmake \
  build-essential \
  wget \
  curl \
  pkg-config \
  libopenblas-dev && \
  rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recursive \
  --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git /llama.cpp

WORKDIR /llama.cpp

# Important:
# - CPU only
# - static binary
RUN cmake \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_HBM=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  . && \
  cmake --build . --target llama-quantize --parallel $(nproc)

RUN pip install --no-cache-dir huggingface_hub

RUN mkdir -p /models

COPY scripts/manage_models.py /scripts/manage_models.py

ENTRYPOINT ["python", "/scripts/manage_models.py"]


# =============================================================================
# Stage 3: Runtime
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  wget \
  curl \
  python3 \
  python3-pip \
  libopenblas0-pthread \
  libgfortran5 && \
  rm -rf /var/lib/apt/lists/*

# ---------------------------
# GPU binaries
# ---------------------------
COPY --from=builder /llama.cpp/bin/llama-server /usr/local/bin/
COPY --from=builder /llama.cpp/bin/llama-cli /usr/local/bin/
COPY --from=builder /llama.cpp/bin/llama-bench /usr/local/bin/

# GPU shared libs needed by server
COPY --from=builder /llama.cpp/bin/libllama*.so* /usr/local/lib/
COPY --from=builder /llama.cpp/bin/libggml*.so* /usr/local/lib/

# ---------------------------
# CPU static quantizer
# ---------------------------
COPY --from=model-manager /llama.cpp/bin/llama-quantize /usr/local/bin/

# OpenBLAS runtime
COPY --from=builder /usr/lib/x86_64-linux-gnu/libopenblas.so* /usr/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH

RUN pip install --no-cache-dir huggingface_hub

ENV MODEL_DIR=/models
RUN mkdir -p /models

COPY scripts/manage_models.py /scripts/manage_models.py
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=120s \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080"]