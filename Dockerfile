# =============================================================================
# Stage 1: Builder (CUDA)
# =============================================================================
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS builder

# Target GPU architecture(s). Default 86 = RTX 3090/3080.
# Override at build time: docker compose build --build-arg CUDA_ARCHS="86;89"
# Use "native" to auto-detect (requires GPU visible at build time).
ARG CUDA_ARCHS=86

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

RUN git clone --depth 1 --branch llama-exp --recursive \
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
  [ ! -f "$BIN/llama-server" ]   && [ -f "$BIN/server" ]   && cp "$BIN/server"   "$BIN/llama-server"   || true

# Strip debug symbols and stage shared libs into a flat dir (~60-80% size reduction)
RUN strip --strip-unneeded /llama.cpp/build/bin/llama-server /llama.cpp/build/bin/llama-quantize 2>/dev/null || true && \
  mkdir -p /staging/lib && \
  find /llama.cpp/build \( -name "libllama*.so*" -o -name "libggml*.so*" -o -name "libmtmd*.so*" \) | \
  xargs -I{} cp {} /staging/lib/ && \
  strip --strip-unneeded /staging/lib/*.so* 2>/dev/null || true


# =============================================================================
# Stage 2: Runtime (server only — assumes models are already in /models volume)
#
# Model preparation is handled entirely by the convert image (stage 3).
# This image contains only llama-server and its runtime dependencies.
# =============================================================================
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04 AS runtime

# Runtime shared libs copied from builder — avoids apt repo issues in the
# minimal CUDA runtime image.  The devel stage already has everything.
COPY --link --from=builder /usr/lib/*/libopenblas*.so* /usr/lib/
COPY --link --from=builder /usr/lib/*/libgomp*.so* /usr/lib/
COPY --link --from=builder /usr/local/cuda/lib64/libcublas*.so* /usr/local/cuda/lib64/
COPY --link --from=builder /usr/local/cuda/lib64/libcublasLt*.so* /usr/local/cuda/lib64/

# curl is the only tool we need from apt (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
  rm -rf /var/lib/apt/lists/*

# Binaries and shared libs from builder (already stripped)
COPY --link --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --link --from=builder /staging/lib/ /opt/llama/lib/

ENV MODEL_DIR=/models \
  LD_LIBRARY_PATH=/opt/llama/lib:/usr/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64 \
  PATH=/usr/local/bin:$PATH

COPY entrypoint.sh /entrypoint.sh
COPY scripts/webui-config.json /etc/llama-server/webui-config.json
COPY scripts/models.ini /etc/llama-server/models.ini
RUN mkdir -p /models && \
  chmod +x /entrypoint.sh && sed -i 's/\r$//' /entrypoint.sh && \
  if [ ! -x "/usr/local/bin/llama-server" ]; then \
  echo "ERROR: /usr/local/bin/llama-server is missing or not executable" && exit 1; \
  fi

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=120s \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]


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
#   docker compose run --rm llama-convert convert-st qwen3.6-35b-a3b --quant TQ2_0
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
COPY --link --from=builder /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

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

# Pull the HF→GGUF conversion script and its support package from the builder.
COPY --link --from=builder /llama.cpp/convert_hf_to_gguf.py /scripts/convert_hf_to_gguf.py
COPY --link --from=builder /llama.cpp/conversion/ /scripts/conversion/
COPY --link --from=builder /llama.cpp/gguf-py/ /scripts/gguf-py/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system --break-system-packages \
  torch --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system --break-system-packages \
  huggingface_hub hf_transfer transformers safetensors \
  sentencepiece accelerate /scripts/gguf-py/

# Copied after pip install so that edits to this script don't bust the pip cache.
COPY scripts/manage_models.py /scripts/manage_models.py

# Enable hf_transfer: Rust-based parallel downloader that ships with
# huggingface_hub.  Provides multi-connection HTTP range downloads for
# substantially higher throughput than the default single-stream requests path.
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Override entrypoint: this image is a CLI tool, not a long-running server.
ENTRYPOINT ["python3", "/scripts/manage_models.py"]
CMD ["--help"]
