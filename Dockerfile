# syntax=docker/dockerfile:1.7
# =============================================================================
# SGLang TurboQuant server image — CUDA 12.6.3 + cuDNN
# Optimized for RTX 3090 (SM86): compiles SM80+SM89+SM90 only.
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
#   docker run --gpus all \
#     -e SGLANG_MODEL_PATH=/models/my-model.gguf \
#     -v /path/to/models:/models \
#     -p 8080:8080 \
#     sglang-turboquant
# =============================================================================

ARG CUDA_VERSION=12.6.3

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
# Defaulting to RTX 3090 (SM86) only — override at build time for other GPUs:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# Common values: "8.0" (A100), "8.6" (RTX 3090), "8.9" (RTX 4090), "9.0" (H100)
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ─── torch-builder ────────────────────────────────────────────────────────────
# Runs in parallel with repo-cloner. Installs PyTorch into a dedicated venv.
# Cache mount: reuses downloaded wheels across rebuilds.
# NOTE: if a new cu126 torch wheel is published with the same version, you must
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
  --index-url https://download.pytorch.org/whl/cu126

# ─── repo-cloner ──────────────────────────────────────────────────────────────
# Runs in parallel with torch-builder (no shared resources).
FROM base AS repo-cloner

RUN git clone --depth 1 \
  --branch feature/turboquant \
  --recursive \
  https://github.com/JEF1056/sglang-turboquant.git \
  /sglang


# ─── sglang-builder ───────────────────────────────────────────────────────────
# Merges outputs of the two parallel stages, then installs sglang on top of
# the pre-built torch venv.
# NOTE: if flashinfer or triton prebuilt wheels for cu126 are unavailable,
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
# No extras: all CUDA optimizations (flashinfer, flash-attn-4, cutlass, tilelang,
# torchao, sgl-kernel, quack-kernels, sgl-deep-gemm, etc.) live in base deps on
# feature/turboquant. The only extras ([all] = diffusion+tracing+http2) are
# irrelevant for LLM inference. PyTorch is already in the venv from torch-builder.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install \
  --no-build-isolation \
  "./python"

# Rebuild sgl-kernel from vendored source for SM86 (RTX 3090) support.
# The PyPI wheel ships SM100-only; source build with ENABLE_BELOW_SM90=ON
# (default on x86_64) produces SM80 + SM86 + SM89 binaries.
# NOTE: triggers nvcc compile. docker-compose.yml pins TORCH_CUDA_ARCH_LIST=8.6
#       to build only SM86 and keep compile time reasonable.
#
# Pre-install build backend into the venv so --no-build-isolation finds them.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install scikit-build-core cmake ninja

# Patch sgl-kernel CMakeLists for RTX 3090 (SM86):
#
# 1. Remove the SM100+ build target (common_ops_sm100_build).
#    sgl-kernel compiles TWO identical shared libraries from the same sources
#    (sm90/ and sm100/); with SM100A=OFF they use the same arch set — pure
#    duplicate work. RTX 3090 will never load the sm100 path anyway.
#    Savings: ~50% of total kernel compile time.
#
# 2. Remove SM100-specific source files from the SOURCES list.
#    These Blackwell-only kernels (mxfp8 blockscaled) produce dead code for
#    SM86 and still take nvcc time even in the sm90/ target.
#
# Uses Python heredoc (requires dockerfile syntax 1.7 + BuildKit).
RUN python3 - <<'PATCH'
import re, pathlib, sys

cmake = pathlib.Path('/sglang/sgl-kernel/CMakeLists.txt')
txt = cmake.read_text()

# 1. Remove SM100+ build target block
before = len(txt)
txt = re.sub(
    r'# =+[^\n]*(?:Common SM100\+|SM100\+ Build)[^\n]*\n.*?install\(TARGETS common_ops_sm100_build[^\n]*\n',
    '',
    txt,
    flags=re.DOTALL | re.IGNORECASE,
)
removed_sm100_target = len(txt) < before
if not removed_sm100_target:
    print("WARNING: SM100+ build target block not found — skipping", file=sys.stderr)

# 2. Remove SM100-specific source files
sm100_sources = [
    '    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu"\n',
    '    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cu"\n',
]
for src in sm100_sources:
    if src in txt:
        txt = txt.replace(src, '')
    else:
        print(f"WARNING: source not found: {src.strip()}", file=sys.stderr)

cmake.write_text(txt)
print(f"Patched CMakeLists.txt — SM100+ target removed: {removed_sm100_target}")
PATCH

# CMAKE_POLICY_VERSION_MINIMUM=3.5: mscclpp sub-project uses cmake_minimum_required
# below 3.5; CMake 4.x removed compatibility. This flag re-enables it.
# SGL_KERNEL_COMPILE_THREADS=2: limits NVCC's internal thread count per job,
# reducing per-process peak memory on top of the parallel job cap.
# CMAKE_BUILD_PARALLEL_LEVEL=2: cap concurrent nvcc jobs; lower if OOM.
# --no-build-isolation: use torch already in venv for ABI consistency.
# --no-deps: skip re-resolving the full dependency graph.
#
# RTX 3090 (SM86) arch reduction flags:
#   ENABLE_BELOW_SM90=ON (default): compiles SM80+SM89 code; SM86 runs SM80 binary.
#   SGL_KERNEL_ENABLE_SM90A=OFF: suppresses SM90a (would auto-enable on CUDA >=12.4).
#   SGL_KERNEL_ENABLE_SM100A=OFF: suppresses SM100a/SM120a (Blackwell).
#   SGL_KERNEL_ENABLE_FA3=OFF: suppresses Flash Attention 3 (also auto-enables SM90a).
# Net result: compiles SM80 + SM89 + SM90 only (vs 8+ arches on CUDA 13.0).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DSGL_KERNEL_COMPILE_THREADS=2 \
    -DSGL_KERNEL_ENABLE_SM90A=OFF \
    -DSGL_KERNEL_ENABLE_SM100A=OFF \
    -DSGL_KERNEL_ENABLE_FA3=OFF" \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
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
COPY --from=sglang-builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
