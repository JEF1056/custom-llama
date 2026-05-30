# syntax=docker/dockerfile:1.7
# =============================================================================
# SGLang TurboQuant server image — CUDA 13.0.1 + cuDNN
# Optimized for RTX 3090 (SM86): compiles SM86 only (no SM90/SM100).
#
# Builds sglang from JEF1056/sglang-turboquant (feature/turboquant)
# using uv for dependency management. Served via entrypoint.sh.
#
# BuildKit is required (enabled by default in Docker 23+).
# Parallel stages: torch-builder and repo-cloner run concurrently.
# Cache mounts avoid re-downloading packages on rebuild.
#
# Build:
#   docker build -t sglang-turboquant .
# Build for other GPU (e.g. H100):
#   docker build --build-arg CUDA_VERSION=13.0.1 --build-arg TORCH_CUDA_ARCH_LIST="9.0" -t sglang-turboquant .
#
# Force-refresh uv/pip cache (e.g. after a new torch release):
#   docker build --no-cache-filter=torch-builder,sglang-builder ...
#
# Run:
#   docker compose up sglang-server
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
  libssl-dev \
  protobuf-compiler \
  libnuma-dev && \
  rm -rf /var/lib/apt/lists/*

# uv — fast Python package manager (pinned for reproducibility)
# https://docs.astral.sh/uv/guides/integration/docker/
COPY --link --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/

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
# Defaulting to RTX 3090 (SM86) only — override at build time for other GPUs:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# Common values: "8.0" (A100), "8.6" (RTX 3090), "8.9" (RTX 4090), "9.0" (H100)
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ─── torch-builder ────────────────────────────────────────────────────────────
# Runs in parallel with repo-cloner. Installs PyTorch into a dedicated venv.
#
# PyTorch 2.11.0 defaults to CUDA 13 (cu130) on PyPI. Pinning here ensures
# uv does not replace the CUDA wheel with a CPU build when sglang installs
# its own dependencies with --no-build-isolation.
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
# Runs in parallel with torch-builder (no shared resources).
FROM base AS repo-cloner

RUN git clone --depth 1 \
  --branch feature/turboquant \
  --recurse-submodules \
  --shallow-submodules \
  https://github.com/JEF1056/sglang-turboquant.git \
  /sglang


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
COPY --link --from=torch-builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

# setuptools_rust is required by sglang's setup.py but not auto-installed when
# --no-build-isolation is used (isolation is off so uv skips build-system deps).
# Installed BEFORE copying sglang source so this layer survives source-only
# rebuilds (torch venv is the only upstream dependency).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
  --mount=type=cache,target=/root/.cargo/git,sharing=locked \
  uv pip install setuptools_rust

# Cloned source tree — placed after stable installs so source changes only
# bust layers from here down.
COPY --link --from=repo-cloner /sglang /sglang

WORKDIR /sglang

# --no-build-isolation: allows the build backend to use the torch already in
# the venv, avoiding a redundant torch download during sglang's native builds.
# Non-editable install: copies everything into site-packages so /sglang is not
# needed at runtime, shaving off the full source + submodule tree.
#
# No extras: all CUDA optimizations (flashinfer, flash-attn-4, cutlass, tilelang,
# torchao, sgl-kernel, quack-kernels, sgl-deep-gemm, etc.) live in base deps on
# feature/turboquant. The only extras ([all] = diffusion+tracing+http2) are
# irrelevant for LLM inference. PyTorch is already in the venv from torch-builder.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
  --mount=type=cache,target=/root/.cargo/git,sharing=locked \
  uv pip install \
  --no-build-isolation \
  "./python"

# kernels==0.15.1 (released 2026-05-29) made revision/version required in
# LayerRepository.__init__. transformers' hub_kernels.py calls it without
# either arg, crashing at startup. Pin to 0.14.1 (last safe release).
# Both 0.14.1 and 0.15.1 require huggingface-hub>=1.10.0, so no cascade.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install "kernels==0.14.1"

# sgl-kernel: compiled from fork source, SM86 only.
# Precompiled wheels target SM100+; SM86 (RTX 3090) requires source build.
# SM90 and SM100 build targets are patched out — SM86 is Ampere and only
# needs the base SM80 binary (fully compatible). Build time: ~10-20 min.

# Build backend for scikit-build-core / cmake.
# cmake is pinned to 3.x: cmake 4.x changed how IMPORTED target include dirs
# propagate, breaking find_package(Torch) header forwarding to build targets.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install scikit-build-core "cmake>=3.27,<4" ninja

# Patch CMakeLists:
#   1. Remove SM100+ build target (Blackwell-only, not needed for SM86)
#   2. Remove SM90 build target (Hopper-only, not needed for SM86) — this
#      also eliminates the header propagation issues that caused build failures
#   3. Inject torch/csrc/api/include path (LibTorch requires two include roots)
RUN python3 - <<'PATCH'
import re, pathlib, subprocess, sys

cmake = pathlib.Path('/sglang/sgl-kernel/CMakeLists.txt')
txt = cmake.read_text()

# ── Remove SM100+ build target ────────────────────────────────────────────
before = len(txt)
txt = re.sub(
    r'# =+[^\n]*(?:Common SM100\+|SM100\+ Build)[^\n]*\n.*?install\(TARGETS common_ops_sm100_build[^\n]*\n',
    '',
    txt,
    flags=re.DOTALL | re.IGNORECASE,
)
if len(txt) == before:
    print("WARNING: SM100+ build target block not found — skipping", file=sys.stderr)

for src in [
    '    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu"\n',
    '    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cu"\n',
]:
    if src in txt:
        txt = txt.replace(src, '')
    else:
        print(f"WARNING: source not found: {src.strip()}", file=sys.stderr)

# ── Remove SM90 build target ──────────────────────────────────────────────
before = len(txt)
txt = re.sub(
    r'# =+[^\n]*SM90[^\n]*\n.*?install\(TARGETS common_ops_sm90_build[^\n]*\n',
    '',
    txt,
    flags=re.DOTALL | re.IGNORECASE,
)
if len(txt) == before:
    print("WARNING: SM90 build target block not found — skipping", file=sys.stderr)

# ── Add missing torch C++ API include path ────────────────────────────────
# LibTorch requires two include paths:
#   1. torch/include/                        (from find_package(Torch))
#   2. torch/include/torch/csrc/api/include/ (missing from cmake targets)
torch_inc = subprocess.check_output(
    ['python3', '-c',
     'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))'],
    text=True).strip()
api_inc = f'{torch_inc}/torch/csrc/api/include'
injection = f'include_directories("{api_inc}")'
marker = 'find_package(Torch REQUIRED)'
if marker in txt and injection not in txt:
    txt = txt.replace(marker, f'{marker}\n{injection}', 1)
    print(f'Injected include_directories for torch/csrc/api/include')
else:
    print('WARNING: could not inject include_directories', file=sys.stderr)

cmake.write_text(txt)
PATCH

# Build sgl-kernel for SM86 only (no SM90/SM100).
# TORCH_CUDA_ARCH_LIST=8.6 (set in base) controls the nvcc arch flags.
# SGL_KERNEL_COMPILE_THREADS=4 limits parallel nvcc jobs inside the build.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DSGL_KERNEL_COMPILE_THREADS=4 \
  -DSGL_KERNEL_ENABLE_SM90A=OFF \
  -DSGL_KERNEL_ENABLE_SM100A=OFF \
  -DSGL_KERNEL_ENABLE_FA3=OFF" \
  CMAKE_BUILD_PARALLEL_LEVEL=4 \
  uv pip install --no-build-isolation --no-deps /sglang/sgl-kernel

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
  ca-certificates \
  libnuma1 && \
  rm -rf /var/lib/apt/lists/*

# Only the venv is needed — non-editable install put everything in site-packages.
COPY --link --from=sglang-builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

COPY --link --chmod=755 entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
