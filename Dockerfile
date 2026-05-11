# =============================================================================
# Stage 1: Builder - Build llama.cpp with CUDA + BLAS
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install build dependencies in a single layer for better caching
# Using --no-install-recommends to minimize image size
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
  libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

# Clone the llama.cpp fork with TurboQuant KV-cache support
# Using --depth 1 to reduce clone size and improve caching
# Pinning to a specific commit would be even better for reproducibility
RUN git clone --depth 1 --recursive --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git /llama.cpp

WORKDIR /llama.cpp

# Build with CUDA support and BLAS acceleration for GEMM operations
# Combining cmake and make into a single RUN layer for better caching
# Note: cmake --parallel is not valid; use cmake --build --parallel for parallel builds
RUN cmake \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_OPENBLAS_INCLUDE_DIR=/usr/include \
  -DGGML_OPENBLAS_LIBRARY=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -DGGML_HBM=OFF \
  -DGGML_CUDA_LINK=static \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda/lib64:/usr/local/nvidia/lib64 -L/usr/local/cuda/lib64/stubs -lcuda" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda/lib64:/usr/local/nvidia/lib64 -L/usr/local/cuda/lib64/stubs -lcuda" \
  . && \
  cmake --build . --parallel $(nproc) --config Release

# Clean up git cache to reduce layer size
RUN rm -rf /llama.cpp/.git

# =============================================================================
# Stage 2: Model Manager - Build llama-quantize (CPU-only) for model conversion
# =============================================================================
FROM python:3.11-slim AS model-manager

# Install build dependencies for llama.cpp in a single layer
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  cmake \
  build-essential \
  wget \
  curl \
  pkg-config \
  libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp with TurboQuant support for llama-quantize
# Using --depth 1 to reduce clone size
RUN git clone --depth 1 --recursive --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git /llama.cpp

WORKDIR /llama.cpp

# Build llama-quantize (CPU-only, no CUDA needed)
# Combining cmake and make into a single RUN layer
# Note: cmake --parallel is not valid; use cmake --build --parallel for parallel builds
RUN cmake \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_OPENBLAS_INCLUDE_DIR=/usr/include \
  -DGGML_OPENBLAS_LIBRARY=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -DGGML_HBM=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  . && \
  cmake --build . --parallel $(nproc) --config Release

# Clean up git cache to reduce layer size
RUN rm -rf /llama.cpp/.git

# Install huggingface_hub for model downloads
# Using --no-deps for reproducibility - but we need typer for huggingface-cli
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
# Using --no-install-recommends to minimize image size
# Explicitly installing BLAS runtime dependencies for OpenBLAS
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  wget \
  curl \
  python3 \
  python3-pip \
  libopenblas0-pthread \
  libgfortran5 \
  && rm -rf /var/lib/apt/lists/*

# Copy llama.cpp binaries from builder stage
# Note: llama.cpp uses in-source build, so binaries are in /llama.cpp/bin/ not /llama.cpp/build/bin/
COPY --from=builder /llama.cpp/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /llama.cpp/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=builder /llama.cpp/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder /llama.cpp/bin/llama-bench /usr/local/bin/llama-bench

# Copy BLAS libraries (for Intel OpenBLAS)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libopenblas.so* /usr/lib/

# Install huggingface-cli for model downloads
# Using --no-deps for reproducibility - but we need typer for huggingface-cli
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
# Improved timeout and start-period for slow-starting llama-server
HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=120s \
  CMD curl -f http://localhost:8080/health || exit 1

# Default command uses entrypoint.sh for flexible configuration
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
