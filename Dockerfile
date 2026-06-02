# syntax=docker/dockerfile:1.7
# SGLang TurboQuant — CUDA 13.0.1, RTX 3090 (SM86).
# Uses prebuilt sglang-kernel wheel (cu130, includes SM86) — no source compile.
# Build: docker build -t sglang-turboquant .
# Run:   docker compose up sglang-server

ARG CUDA_VERSION=13.0.1

# ─── base ─────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git curl ca-certificates \
  python3 python3-dev python3-venv \
  build-essential ninja-build pkg-config \
  libssl-dev protobuf-compiler libnuma-dev && \
  rm -rf /var/lib/apt/lists/*

COPY --link --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PYTHON_DOWNLOADS=never

ENV PATH="/root/.cargo/bin:$PATH"

RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal \
  && rustc --version && cargo --version

ENV CUDA_HOME=/usr/local/cuda

# ─── torch-builder ────────────────────────────────────────────────────────────
# Installs PyTorch cu130 into a venv; pinned so sglang's --no-build-isolation
# install doesn't replace it with a CPU wheel.
FROM base AS torch-builder

RUN uv venv /opt/venv --python python3

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install \
  "torch==2.11.0+cu130" \
  "torchvision==0.26.0+cu130" \
  "torchaudio==2.11.0+cu130" \
  --index-url https://download.pytorch.org/whl/cu130

# ─── repo-cloner ──────────────────────────────────────────────────────────────
FROM base AS repo-cloner

# Pass --build-arg CACHEBUST=$(date +%s) to force a fresh clone.
ARG CACHEBUST=1

RUN git clone --depth 1 \
  --branch feature/turboquant-rebase-main \
  --recurse-submodules \
  --shallow-submodules \
  https://github.com/JEF1056/sglang-turboquant.git \
  /sglang

# ─── sglang-builder ───────────────────────────────────────────────────────────
FROM base AS sglang-builder

COPY --link --from=torch-builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

# setuptools_rust needed by sglang's setup.py; must be present before
# --no-build-isolation install skips build-system deps.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
  --mount=type=cache,target=/root/.cargo/git,sharing=locked \
  uv pip install setuptools_rust

# cmake pinned to 3.x — cmake 4.x broke find_package(Torch) include propagation.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install scikit-build-core "cmake>=3.27,<4" ninja

COPY --link --from=repo-cloner /sglang /sglang

WORKDIR /sglang

# Install prebuilt sglang-kernel (cu130, SM86 included) — no source compile.
# Pinned to the version declared in python/pyproject.toml (sglang-kernel==0.4.3).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install --force-reinstall "sglang-kernel==0.4.3" \
  --index-url https://docs.sglang.ai/whl/cu130/

# Non-editable install with --no-build-isolation so torch stays as the
# pinned cu130 wheel; sglang-kernel is already satisfied above.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
  --mount=type=cache,target=/root/.cargo/git,sharing=locked \
  uv pip install --no-build-isolation "./python"

# kernels==0.15.1 broke transformers (hub_kernels.py calls LayerRepository
# without required args). Pin to last safe release.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install "kernels==0.14.1"

# ─── venv-trim ────────────────────────────────────────────────────────────────
# Strip test suites, build-time packages, unused torch components, and debug
# symbols from the venv before copying to the runtime stage.
RUN SITE=/opt/venv/lib/python3.12/site-packages && \
  # Test suites and benchmarks — never needed at inference time
  find /opt/venv -type d \( -name "tests" -o -name "test" -o -name "benchmarks" \) \
  -exec rm -rf {} + 2>/dev/null || true && \
  # torch C++ headers — only needed when compiling CUDA extensions
  rm -rf "$SITE/torch/include" && \
  # torch bundled test utilities
  rm -rf "$SITE/torch/test" && \
  # Build-time packages — cmake, ninja, scikit-build-core, setuptools_rust
  rm -rf \
  "$SITE"/cmake* "$SITE"/CMake* \
  "$SITE"/ninja* \
  "$SITE"/scikit_build_core* \
  "$SITE"/setuptools_rust* \
  "$SITE"/_setuptools_rust* && \
  # CMake metadata / pkg-config files (build-time only)
  find /opt/venv -type d -name "cmake" -exec rm -rf {} + 2>/dev/null || true && \
  find /opt/venv -name "*.pdb" -delete 2>/dev/null || true && \
  # Strip debug symbols from all shared libraries (~20-30% size reduction on .so files)
  find /opt/venv -name '*.so' -exec strip --strip-debug {} \; 2>/dev/null || true

# ─── runtime ──────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python3 python3-dev ca-certificates libnuma1 \
  ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev && \
  rm -rf /var/lib/apt/lists/*

# Strip CUDA devel-only artifacts not needed for inference.
# Saves ~3–5 GB: static libs, samples, docs, and profiling tools.
RUN find /usr/local/cuda/lib64 -name '*.a' -delete 2>/dev/null || true && \
  find /usr/lib -name 'libcudnn_*_static.a' -delete 2>/dev/null || true && \
  rm -rf \
  /usr/local/cuda/samples \
  /usr/local/cuda/extras \
  /usr/local/cuda/doc \
  /usr/local/cuda/nsight-compute-* \
  /usr/local/cuda/nsight-systems-* \
  /usr/local/cuda/NsightCompute* \
  /usr/local/cuda/NsightSystems* \
  /usr/local/cuda/compute-sanitizer

COPY --link --from=sglang-builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

COPY --link --chmod=755 entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
