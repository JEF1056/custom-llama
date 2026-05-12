# =============================================================================
# Stage 1: Builder (CUDA)
# =============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  cmake \
  build-essential \
  libopenblas-dev \
  libssl-dev \
  pkg-config && \
  rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recursive \
  https://github.com/TheTom/llama-cpp-turboquant.git /llama.cpp

WORKDIR /llama.cpp

# --allow-shlib-undefined is required because libcuda.so.1 is a host driver
# stub injected at runtime by the NVIDIA container runtime — it is intentionally
# absent at build time. Without this flag, the linker fails on every tool binary.
RUN cmake \
  -B build \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_HBM=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS_CXX=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
  . && \
  cmake --build build --parallel $(nproc)

# Confirm what was built
RUN echo "=== Built binaries ===" && find /llama.cpp/build/bin -type f | sort

# Normalize binary names using cp (not ln -sf — symlinks break across COPY --from)
RUN set -e; \
  BIN=/llama.cpp/build/bin; \
  [ ! -f "$BIN/llama-quantize" ] && [ -f "$BIN/quantize" ] && cp "$BIN/quantize" "$BIN/llama-quantize" || true; \
  [ ! -f "$BIN/llama-server" ]   && [ -f "$BIN/server" ]   && cp "$BIN/server"   "$BIN/llama-server"   || true; \
  echo "=== Final bin contents ===" && ls -la "$BIN/"

# Stage .so files into a flat dir — avoids glob issues with COPY --from
RUN mkdir -p /staging/lib && \
  find /llama.cpp/build -name "libllama*.so*" -o -name "libggml*.so*" | \
  xargs -I{} cp {} /staging/lib/


# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  wget \
  python3 \
  python3-pip \
  libopenblas0-pthread \
  libssl3 \
  libcublas-12-4 \
  libgomp1 && \
  rm -rf /var/lib/apt/lists/*

ENV MODEL_DIR=/models
RUN mkdir -p /models

# Copy binaries (real files, not symlinks)
COPY --from=builder /llama.cpp/build/bin/ /usr/local/bin/

# Copy .so files from staging dir
RUN mkdir -p /opt/llama/lib
COPY --from=builder /staging/lib/ /opt/llama/lib/

# Copy OpenBLAS without hardcoding the arch path
RUN --mount=from=builder,target=/builder \
  find /builder/usr/lib -name "libopenblas*.so*" -exec cp {} /usr/lib/ \;

ENV LD_LIBRARY_PATH=/opt/llama/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH

RUN pip install --no-cache-dir huggingface_hub

COPY scripts/manage_models.py /scripts/manage_models.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Fail fast at build time if critical binaries are missing
RUN for bin in llama-server llama-quantize; do \
  if [ ! -x "/usr/local/bin/$bin" ]; then \
  echo "ERROR: /usr/local/bin/$bin is missing or not executable" && exit 1; \
  fi; \
  done

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=120s \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080"]