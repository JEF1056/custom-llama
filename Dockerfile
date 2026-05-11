# =============================================================================
# Stage 1: Builder - Build llama.cpp with CUDA + BLAS
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    wget \
    curl \
    python3 \
    python3-pip \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the llama.cpp fork with TurboQuant KV-cache support
# https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache
# This fork adds full TurboQuant KV-cache optimization for efficient inference
# with TQ1_0 (1-bit) and TQ2_0 (2-bit) quantized models
RUN git clone --recursive --branch feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git /llama.cpp
WORKDIR /llama.cpp

# Build with CUDA support and BLAS acceleration for GEMM operations
RUN cmake -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_OPENBLAS_INCLUDE_DIR=/usr/include \
    -DGGML_OPENBLAS_LIBRARY=/usr/lib/x86_64-linux-gnu/libopenblas.so \
    -DGGML_HBM=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    .
RUN make -j$(nproc)

# =============================================================================
# Stage 2: Model Manager - Build llama-quantize (CPU-only) for model conversion
# =============================================================================
FROM python:3.11-slim AS model-manager

# Install build dependencies for llama.cpp
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    wget \
    curl \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp with TurboQuant support for llama-quantize
RUN git clone --recursive --branch feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git /llama.cpp
WORKDIR /llama.cpp

# Build llama-quantize (CPU-only, no CUDA needed)
RUN cmake -DBUILD_SHARED_LIBS=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_OPENBLAS_INCLUDE_DIR=/usr/include \
    -DGGML_OPENBLAS_LIBRARY=/usr/lib/x86_64-linux-gnu/libopenblas.so \
    -DGGML_HBM=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    .
RUN make -j$(nproc)

# Install huggingface_hub for model downloads
RUN pip install --no-cache-dir huggingface_hub

# Create models directory
RUN mkdir -p /models

# Copy model management script
COPY scripts/manage_models.py /scripts/manage_models.py

# Default entrypoint for model-manager stage
ENTRYPOINT ["python", "/scripts/manage_models.py"]

# =============================================================================
# Stage 3: Runtime - Final image with llama-server and all tools
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy llama.cpp binaries from builder stage
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=builder /llama.cpp/build/bin/llama-convert-hf-to-gguf /usr/local/bin/llama-convert-hf-to-gguf
COPY --from=builder /llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder /llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench

# Copy BLAS libraries (for Intel OpenBLAS)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libopenblas.so* /usr/lib/


# Install huggingface-cli for model downloads
RUN pip install --no-cache-dir huggingface_hub

# Set model directory
ENV MODEL_DIR=/models
RUN mkdir -p /models

# Copy model management script
COPY scripts/manage_models.py /scripts/manage_models.py

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the default llama-server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command uses entrypoint.sh for flexible configuration
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
