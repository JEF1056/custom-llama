# =============================================================================
# Stage 1: Builder (CUDA)
# =============================================================================
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS builder

# Target GPU architecture(s). Default 86 = RTX 3090/3080.
# Override at build time: docker compose build --build-arg CUDA_ARCHS="86;89"
# Use "native" to auto-detect (requires GPU visible at build time).
ARG CUDA_ARCHS=86

# Enable synchronous CUDA kernel error checking (VERY slow — debugging only).
# docker compose build --build-arg CUDA_SYNC_DEBUG=1
ARG CUDA_SYNC_DEBUG=0

RUN rm -rf /var/lib/apt/lists/* && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  cmake \
  ninja-build \
  build-essential \
  libopenblas-dev \
  libssl-dev \
  pkg-config && \
  rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch llama-next --recursive \
  https://github.com/JEF1056/llama-cpp-turboquant.git /llama.cpp

WORKDIR /llama.cpp

# --allow-shlib-undefined is required because libcuda.so.1 is a host driver
# stub injected at runtime by the NVIDIA container runtime — it is intentionally
# absent at build time. Without this flag, the linker fails on every tool binary.
RUN cmake \
  -B build -G Ninja \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_HBM=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS_CXX=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  $([ "${CUDA_SYNC_DEBUG}" = "1" ] && echo "-DCMAKE_CUDA_FLAGS=-DGGML_CUDA_SYNC_DEBUG -DCMAKE_CXX_FLAGS=-DGGML_CUDA_SYNC_DEBUG" || true) \
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
  [ ! -f "$BIN/llama-server" ]   && [ -f "$BIN/server" ]   && cp "$BIN/server"   "$BIN/llama-server"   || true

# Strip debug symbols and stage shared libs into a flat dir (~60-80% size reduction)
RUN strip --strip-unneeded /llama.cpp/build/bin/llama-server /llama.cpp/build/bin/llama-quantize 2>/dev/null || true && \
  mkdir -p /staging/lib && \
  find /llama.cpp/build \( -name "libllama*.so*" -o -name "libggml*.so*" -o -name "libmtmd*.so*" \) | \
  xargs -I{} cp {} /staging/lib/ && \
  strip --strip-unneeded /staging/lib/*.so* 2>/dev/null || true


# =============================================================================
# Stage 2: Base (shared runtime foundation for both runtime and convert)
#
# Contains: CUDA runtime image, all shared libs from builder, staged llama libs,
# and common environment variables.  Neither runtime nor convert depend on each
# other — both inherit from this stage only.
# =============================================================================
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04 AS base

# Runtime shared libs copied from builder — avoids apt repo issues in the
# minimal CUDA runtime image.  The devel stage already has everything.
COPY --link --from=builder /usr/lib/*/libopenblas*.so* /usr/lib/
COPY --link --from=builder /usr/lib/*/libgfortran*.so* /usr/lib/
COPY --link --from=builder /usr/lib/*/libgomp*.so* /usr/lib/
COPY --link --from=builder /usr/local/cuda/lib64/libcublas*.so* /usr/local/cuda/lib64/
COPY --link --from=builder /usr/local/cuda/lib64/libcublasLt*.so* /usr/local/cuda/lib64/

# Shared llama libs (used by both llama-server and llama-quantize)
COPY --link --from=builder /staging/lib/ /opt/llama/lib/

ENV MODEL_DIR=/models \
  LD_LIBRARY_PATH=/opt/llama/lib:/usr/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64 \
  PATH=/usr/local/bin:$PATH


# =============================================================================
# Stage 3: Runtime (server only — assumes models are already in /models volume)
#
# Model preparation is handled entirely by the convert image (stage 4).
# This image contains only llama-server and its runtime dependencies.
# =============================================================================
FROM base AS runtime

# curl is the only tool we need from apt (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
  rm -rf /var/lib/apt/lists/*

# llama-server binary from builder (already stripped)
COPY --link --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

COPY entrypoint.sh /entrypoint.sh
# webui-config.json and models.ini are NOT baked in — they are mounted as
# read-only bind mounts in docker-compose.yml so edits take effect on
# `docker compose up --force-recreate` without a rebuild.
RUN mkdir -p /models /etc/llama-server && \
  chmod +x /entrypoint.sh && sed -i 's/\r$//' /entrypoint.sh && \
  if [ ! -x "/usr/local/bin/llama-server" ]; then \
  echo "ERROR: /usr/local/bin/llama-server is missing or not executable" && exit 1; \
  fi

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=120s \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]


# =============================================================================
# Stage 4: Convert (model preparation — download, quantize, convert safetensors)
#
# Inherits from base (NOT runtime) — independent of the server image so changes
# to entrypoint, configs, or healthcheck do not invalidate this stage.
#
# Run before starting the server to prepare models in the shared /models volume.
#
# Examples:
#   # Download a pre-built quant:
#   docker compose run --rm llama-convert download qwen3.5-27b --quant Q4_K_M
#
#   # Download + quantize (fp16 GGUF source on HF):
#   docker compose run --rm llama-convert download qwopus3.6-27b --quant TQ2_0
#
#   # Convert safetensors → fp16 GGUF → quant (no pre-built GGUF exists):
#   docker compose run --rm llama-convert convert-st qwen3.6-35b-a3b --quant TQ2_0
#
# CUDA torch for GPU-accelerated TriAttention calibration (forward pass).
# Quantization (llama-quantize) remains CPU-bound but benefits from the real
# libcuda.so.1 injected at runtime rather than the build-time stub.
# =============================================================================
FROM base AS convert

# libcuda.so.1 is injected at runtime by the NVIDIA container runtime (gpus: all
# in compose).  No build-time stub needed — adding the stubs dir to
# LD_LIBRARY_PATH would shadow the real driver and cause
# torch.cuda.is_available() to return False.

# Model management tools not present in the runtime image
COPY --link --from=builder /llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize

# Python stack for model download and HF→GGUF conversion
# uv: Rust-based pip replacement — 10-100x faster dependency resolution & install.
COPY --link --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN rm -rf /var/lib/apt/lists/* && \
  apt-get update && \
  apt-get install -y --no-install-recommends python3 \
  aria2 && \
  rm -rf /var/lib/apt/lists/*

# Install Python deps in three layers ordered from slowest-changing to fastest:
#   1. torch  — large, pinned CUDA index; almost never changes
#   2. HF stack — changes occasionally with upstream releases
#   3. gguf-py — changes only when llama.cpp is rebuilt (from builder stage)
#
# --no-compile: skip .pyc bytecode generation (~200MB saved for torch alone).
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system --break-system-packages --no-compile \
  torch --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system --break-system-packages --no-compile \
  huggingface_hub hf_transfer transformers safetensors \
  sentencepiece accelerate

# Pull the HF→GGUF conversion script and its support package from the builder.
# Done after the HF stack install so that a llama.cpp rebuild only busts this
# layer and below — not the large torch/HF layers above.
COPY --link --from=builder /llama.cpp/convert_hf_to_gguf.py /scripts/convert_hf_to_gguf.py
COPY --link --from=builder /llama.cpp/conversion/ /scripts/conversion/
COPY --link --from=builder /llama.cpp/gguf-py/ /scripts/gguf-py/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system --break-system-packages --no-compile /scripts/gguf-py/ && \
  find /usr/lib/python3 /usr/local/lib/python3* -type d -name __pycache__ \
  -exec rm -rf {} + 2>/dev/null || true

# Copied last so edits to these scripts never bust any pip cache layer.
COPY scripts/manage_models.py /scripts/manage_models.py
COPY scripts/triattention_calibrate.py /scripts/triattention_calibrate.py
COPY scripts/triattention_common.py /scripts/triattention_common.py

# Enable hf_transfer: Rust-based parallel downloader that ships with
# huggingface_hub.  Provides multi-connection HTTP range downloads for
# substantially higher throughput than the default single-stream requests path.
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Override entrypoint: this image is a CLI tool, not a long-running server.
ENTRYPOINT ["python3", "/scripts/manage_models.py"]
CMD ["--help"]
