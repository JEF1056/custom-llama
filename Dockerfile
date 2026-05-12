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
  find /llama.cpp/build \( -name "libllama*.so*" -o -name "libggml*.so*" -o -name "libmtmd*.so*" \) | \
  xargs -I{} cp {} /staging/lib/


# =============================================================================
# Stage 2: Runtime (server only — assumes models are already in /models volume)
#
# Model preparation is handled entirely by the convert image (stage 3).
# This image contains only llama-server and its runtime dependencies.
# =============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  bash \
  curl \
  libopenblas0-pthread \
  libssl3 \
  libcublas-12-4 \
  libgomp1 && \
  rm -rf /var/lib/apt/lists/*

ENV MODEL_DIR=/models
RUN mkdir -p /models

# llama-server only — quantize and model-management tools live in the convert image
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Copy .so files from staging dir
RUN mkdir -p /opt/llama/lib
COPY --from=builder /staging/lib/ /opt/llama/lib/

# Copy OpenBLAS without hardcoding the arch path
RUN --mount=from=builder,target=/builder \
  find /builder/usr/lib -name "libopenblas*.so*" -exec cp {} /usr/lib/ \;

ENV LD_LIBRARY_PATH=/opt/llama/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && sed -i 's/\r$//' /entrypoint.sh && \
  if [ ! -x "/usr/local/bin/llama-server" ]; then \
  echo "ERROR: /usr/local/bin/llama-server is missing or not executable" && exit 1; \
  fi

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=120s \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080"]


# =============================================================================
# Stage 3: Convert (model preparation — download, quantize, convert safetensors)
#
# Run before starting the server to prepare models in the shared /models volume.
#
# Examples:
#   # Download a pre-built quant:
#   docker compose run --rm llama-convert download qwen3.5-27b --quant Q4_K_M
#
#   # Download + quantize (fp16 GGUF source on HF):
#   docker compose run --rm llama-convert download qwen3.6-27b --quant TQ2_0
#
#   # Convert safetensors → fp16 GGUF → quant (no pre-built GGUF exists):
#   docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0
#
# CPU-only torch keeps the image ~3 GB lighter than CUDA torch.
# Conversion is memory-bound, not compute-bound — CPU is fine.
# =============================================================================
FROM runtime AS convert

# llama-quantize is linked against libcuda.so.1, which is normally injected by
# the NVIDIA container runtime from the host driver.  llama-convert runs without
# GPU access, so the injection never happens and the binary fails to start.
# Copying the build-time stub satisfies the dynamic linker; quantization itself
# is purely CPU-bound so nothing ever calls into the real driver.
COPY --from=builder /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Model management tools not present in the runtime image
COPY --from=builder /llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize

# Python stack for model download and HF→GGUF conversion
RUN apt-get update && \
  apt-get install -y --no-install-recommends python3 python3-pip \
  aria2 && \
  rm -rf /var/lib/apt/lists/*

# Pull the HF→GGUF conversion script and its support package from the builder.
COPY --from=builder /llama.cpp/convert_hf_to_gguf.py /scripts/convert_hf_to_gguf.py
COPY --from=builder /llama.cpp/gguf-py/ /scripts/gguf-py/

RUN python3 -m pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cpu && \
  python3 -m pip install --no-cache-dir \
  huggingface_hub \
  hf_transfer \
  transformers \
  safetensors \
  sentencepiece \
  accelerate \
  /scripts/gguf-py/

# Copied after pip install so that edits to this script don't bust the pip cache.
COPY scripts/manage_models.py /scripts/manage_models.py

# Enable hf_transfer: Rust-based parallel downloader that ships with
# huggingface_hub.  Provides multi-connection HTTP range downloads for
# substantially higher throughput than the default single-stream requests path.
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Override entrypoint: this image is a CLI tool, not a long-running server.
ENTRYPOINT ["python3", "/scripts/manage_models.py"]
CMD ["--help"]
