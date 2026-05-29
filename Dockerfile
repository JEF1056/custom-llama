# syntax=docker/dockerfile:1.7
# =============================================================================
# SGLang TurboQuant server image — CUDA 13.0.1 + cuDNN
#
# Builds sglang from JEF1056/sglang-turboquant (feature/turboquant)
# using uv for dependency management. Served via entrypoint.sh.
#
# BuildKit is required (enabled by default in Docker 23+).
# Parallel stages: torch-builder and repo-cloner run concurrently.
# Cache mounts avoid re-downloading packages on rebuild.
#
# Build:
#   docker build --build-arg CUDA_VERSION=13.0.1 -t sglang-turboquant .
#
# Force-refresh uv/pip cache (e.g. after a new torch release):
#   docker build --no-cache-filter=torch-builder,sglang-builder ...
#
# Run:
#   docker run --gpus all \
#     -e SGLANG_MODEL_PATH=/models/my-model.gguf \
#     -v /path/to/models:/models \
#     -p 8080:8080 \
#     sglang-turboquant
# =============================================================================

ARG CUDA_VERSION=13.0.1

# ─── base: system packages + uv ──────────────────────────────────────────────
# Shared parent for all build stages; deduped automatically by BuildKit.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  curl \
  ca-certificates \
  python3 \
  python3-dev \
  python3-venv \
  build-essential \
  ninja-build \
  pkg-config \
  libssl-dev && \
  rm -rf /var/lib/apt/lists/*

# uv — fast Python package manager (pinned for reproducibility)
# https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PYTHON_DOWNLOADS=never

# Rust toolchain — required by sglang's setup.py (setuptools_rust extensions).
# PATH must be set before the RUN so the verification commands can find rustc.
ENV PATH="/root/.cargo/bin:$PATH"

RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal \
  && rustc --version && cargo --version

# CUDA_HOME: not set by the base image as an env var, but required by many
# build systems (flashinfer, triton native builds, cmake-based extensions).
# TORCH_CUDA_ARCH_LIST: limit native extension compilation to your GPU's SM.
# Override at build time: --build-arg TORCH_CUDA_ARCH_LIST="9.0"
# Common values: "8.0" (A100), "8.6" (RTX 3090), "8.9" (RTX 4090), "9.0" (H100)
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ─── torch-builder ────────────────────────────────────────────────────────────
# Runs in parallel with repo-cloner. Installs PyTorch into a dedicated venv.
# Cache mount: reuses downloaded wheels across rebuilds.
# NOTE: if a new cu130 torch wheel is published with the same version, you must
# invalidate this layer manually (--no-cache-filter=torch-builder) to pick it up.
FROM base AS torch-builder

RUN uv venv /opt/venv --python python3

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install \
  torch \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

# ─── repo-cloner ──────────────────────────────────────────────────────────────
# Runs in parallel with torch-builder (no shared resources).
FROM base AS repo-cloner

RUN git clone --depth 1 \
  --branch feature/turboquant \
  --recursive \
  https://github.com/JEF1056/sglang-turboquant.git \
  /sglang

# InternS2PreviewConfig is imported in hf_transformers/common.py (import block
# and _CONFIG_REGISTRY list) but is missing from configs/__init__.py — broken
# merge. Remove both occurrences until the class is added to the branch.
RUN sed -i '/InternS2PreviewConfig,/d' \
  /sglang/python/sglang/srt/utils/hf_transformers/common.py

# ─── sglang-builder ───────────────────────────────────────────────────────────
# Merges outputs of the two parallel stages, then installs sglang on top of
# the pre-built torch venv.
# NOTE: if flashinfer or triton prebuilt wheels for cu130 are unavailable,
# the install will fail. Check:
#   https://github.com/flashinfer-ai/flashinfer/releases
#   https://github.com/triton-lang/triton/releases
# and add --extra-index-url entries as needed.
FROM base AS sglang-builder

# Venv with torch already installed
COPY --from=torch-builder /opt/venv /opt/venv
# Cloned source tree
COPY --from=repo-cloner /sglang /sglang

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

WORKDIR /sglang

# setuptools_rust is required by sglang's setup.py but not auto-installed when
# --no-build-isolation is used (isolation is off so uv skips build-system deps).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install setuptools_rust

# --no-build-isolation: allows the build backend to use the torch already in
# the venv, avoiding a redundant torch download during sglang's native builds.
# Non-editable install: copies everything into site-packages so /sglang is not
# needed at runtime, shaving off the full source + submodule tree.
#
# [runtime_common] instead of [all]: the fork maps [all] → [all_hip] (AMD ROCm)
# which installs HIP-specific wheels (petit_kernel, wave-lang) incompatible with
# CUDA. [runtime_common] contains all core LLM serving deps without the HIP extras.
# PyTorch is already in the venv from the torch-builder stage.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install \
  --no-build-isolation \
  "./python[runtime_common]"

# ─── runtime ──────────────────────────────────────────────────────────────────
# cudnn-runtime (~8 GB) instead of cudnn-devel (~15 GB) — saves ~5-8 GB.
# Triton 3.x ships its own LLVM/PTX backend and does NOT need system nvcc.
# Flashinfer ships prebuilt kernels. Neither requires the full CUDA dev toolkit.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python3 \
  python3-dev \
  ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# Only the venv is needed — non-editable install put everything in site-packages.
COPY --from=sglang-builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
