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
#   docker compose up sglang-server
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
# Versions are pinned to match what sglang's pyproject.toml requires.
# sglang installs deps with --no-build-isolation; if torch is already at the
# exact version required (local +cu126 suffix satisfies the bare version per
# PEP 440), uv leaves it alone and the CUDA variant stays in the venv.
# Without this pin, uv replaces +cu126 torch with a CPU wheel from PyPI,
# causing sgl-kernel to fail: ATen/cuda headers are absent in CPU torch.
FROM base AS torch-builder

RUN uv venv /opt/venv --python python3

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install \
  "torch==2.11.0+cu126" \
  "torchvision==0.26.0+cu126" \
  "torchaudio==2.11.0+cu126" \
  --index-url https://download.pytorch.org/whl/cu126

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
# NOTE: if flashinfer or triton prebuilt wheels for cu126 are unavailable,
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

# sgl-kernel: compiled from fork source.
# Precompiled wheels (PyPI and turboquant index) are sm100-only; SM86 requires source build.
# Compiles SM80+SM89+SM90 (no SM100+/FA3); SM86 runs the SM80 binary.
# First build ~30-60 min; subsequent rebuilds are cached.

# Build backend for scikit-build-core / cmake.
# cmake is pinned to 3.x: cmake 4.x changed how IMPORTED target include dirs
# propagate, breaking find_package(Torch) header forwarding to the sm90 build
# target (ATen/cuda/CUDAContext.h not found despite torch being installed).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install scikit-build-core "cmake>=3.27,<4" ninja

# Patch CMakeLists: remove SM100+ build target and Blackwell-only source files.
# Eliminates ~50% of compile time on SM86 (RTX 3090).
RUN python3 - <<'PATCH'
import re, pathlib, subprocess, sys

cmake = pathlib.Path('/sglang/sgl-kernel/CMakeLists.txt')
txt = cmake.read_text()

# ── Remove SM100+ build target (saves ~50% compile time on SM86) ──────────
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

# ── Add missing torch C++ API include path ─────────────────────────────────
# LibTorch requires two include paths:
#   1. torch/include/               (added by find_package(Torch))
#   2. torch/include/torch/csrc/api/include/  (NOT added by the sm90 target)
# Without (2), <torch/types.h> and other C++ frontend headers are not found,
# causing "namespace torch has no member Tensor" errors.
torch_inc = subprocess.check_output(
    ['python3', '-c',
     'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))'],
    text=True).strip()
api_inc = f'{torch_inc}/torch/csrc/api/include'
injection = f'include_directories("{api_inc}")'
# Insert once, right after find_package(Torch REQUIRED)
marker = 'find_package(Torch REQUIRED)'
if marker in txt and injection not in txt:
    txt = txt.replace(marker, f'{marker}\n{injection}', 1)
    print(f'Injected include_directories for torch/csrc/api/include')
else:
    print('WARNING: could not inject include_directories — marker not found or already present',
          file=sys.stderr)

cmake.write_text(txt)
PATCH

# torch/all.h compatibility shim.
# PyTorch 2.11.0 does not ship torch/all.h; sgl-kernel sources include it
# directly. Create it when missing so compilation succeeds without patching
# each source file.
RUN python3 - <<'SHIM'
import os, torch
inc = os.path.join(os.path.dirname(torch.__file__), 'include')
dst = os.path.join(inc, 'torch', 'all.h')
# The shim must be self-contained. Forwarding to torch/torch.h creates a
# circular dependency: torch.h itself includes <torch/all.h>, which hits
# our shim's #pragma once guard, leaving torch:: namespace empty.
# Instead, build the torch:: namespace directly from ATen/c10 headers
# (which have stable locations) and enumerate the aliases that sgl-kernel
# CUDA sources actually require.
content = """\
#pragma once
// Compatibility shim: torch/all.h absent/relocated in PyTorch 2.11+.
// Self-contained — no forwarding to any header that re-includes torch/all.h
// (which would hit the #pragma once guard and leave torch:: empty).
// Uses only the canonical c10 constant names stable since PyTorch 1.x.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Device.h>

namespace torch {
  // Core type aliases
  using Tensor        = at::Tensor;
  using TensorOptions = c10::TensorOptions;
  using ScalarType    = c10::ScalarType;
  using Device        = c10::Device;
  using DeviceType    = c10::DeviceType;
  using Generator     = at::Generator;

  // Tensor factory functions
  using at::empty;
  using at::zeros;
  using at::ones;
  using at::full;
  using at::cat;
  using at::stack;
  using at::arange;

  // Canonical ScalarType constants (stable in c10 since PyTorch 1.x).
  // Avoid kUInt8/kInt8/kInt16/kInt32/kFloat16 etc. — those newer aliases
  // may not exist in all 2.x builds.
  constexpr auto kByte     = c10::kByte;
  constexpr auto kChar     = c10::kChar;
  constexpr auto kShort    = c10::kShort;
  constexpr auto kInt      = c10::kInt;
  constexpr auto kLong     = c10::kLong;
  constexpr auto kHalf     = c10::kHalf;
  constexpr auto kFloat    = c10::kFloat;
  constexpr auto kDouble   = c10::kDouble;
  constexpr auto kBFloat16 = c10::kBFloat16;

  // Device type constants
  constexpr auto kCPU  = c10::kCPU;
  constexpr auto kCUDA = c10::kCUDA;
}  // namespace torch
"""
with open(dst, 'w') as f:
    f.write(content)
print(f'Wrote self-contained torch/all.h shim at {dst}')
SHIM

# Build flags: SM80+SM89+SM90, no SM90a/SM100a/FA3.
# CMAKE_CUDA_FLAGS: the sgl-kernel CMakeLists finds torch (libtorch.so) via
# find_package(Torch) but does not propagate TORCH_INCLUDE_DIRS to the sm90
# target's compile command. Injecting the path via CMAKE_CUDA_FLAGS ensures
# all nvcc invocations get ATen/cuda/ and c10/cuda/ headers.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  TORCH_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))") && \
  CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CUDA_FLAGS=-I${TORCH_INC} \
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
