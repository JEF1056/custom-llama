# syntax=docker/dockerfile:1.7
# SGLang TurboQuant — CUDA 13.0.1, RTX 3090 (SM86 only, no SM90/SM100).
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

# Override at build time for other GPUs, e.g. --build-arg TORCH_CUDA_ARCH_LIST="9.0"
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

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

RUN git clone --depth 1 \
  --branch feature/turboquant \
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

COPY --link --from=repo-cloner /sglang /sglang

WORKDIR /sglang

# Non-editable install with --no-build-isolation so torch stays as the
# pinned cu130 wheel; all CUDA deps live in the branch's base requirements.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
  --mount=type=cache,target=/root/.cargo/git,sharing=locked \
  uv pip install --no-build-isolation "./python"

# kernels==0.15.1 broke transformers (hub_kernels.py calls LayerRepository
# without required args). Pin to last safe release.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install "kernels==0.14.1"

# cmake pinned to 3.x — cmake 4.x broke find_package(Torch) include propagation.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv pip install scikit-build-core "cmake>=3.27,<4" ninja

# Patch CMakeLists: remove SM90 + SM100 targets (Hopper/Blackwell, not needed
# for SM86), and inject the missing torch/csrc/api/include path.
RUN python3 - <<'PATCH'
import re, pathlib, subprocess, sys

cmake = pathlib.Path('/sglang/sgl-kernel/CMakeLists.txt')
txt = cmake.read_text()

for pattern, label in [
    (r'# =+[^\n]*(?:Common SM100\+|SM100\+ Build)[^\n]*\n.*?install\(TARGETS common_ops_sm100_build[^\n]*\n', 'SM100+'),
    (r'# =+[^\n]*SM90[^\n]*\n.*?install\(TARGETS common_ops_sm90_build[^\n]*\n', 'SM90'),
]:
    before = len(txt)
    txt = re.sub(pattern, '', txt, flags=re.DOTALL | re.IGNORECASE)
    if len(txt) == before:
        print(f'WARNING: {label} build target not found', file=sys.stderr)

for src in [
    '    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu"\n',
    '    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cu"\n',
]:
    if src in txt:
        txt = txt.replace(src, '')

torch_inc = subprocess.check_output(
    ['python3', '-c', 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))'],
    text=True).strip()
api_inc = f'{torch_inc}/torch/csrc/api/include'
injection = f'include_directories("{torch_inc}")\ninclude_directories("{api_inc}")'
marker = 'find_package(Torch REQUIRED)'
if marker in txt and f'include_directories("{torch_inc}")' not in txt:
    txt = txt.replace(marker, f'{marker}\n{injection}', 1)

cmake.write_text(txt)
PATCH

# SM86 only; TORCH_CUDA_ARCH_LIST=8.6 (from base) drives nvcc arch selection.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DSGL_KERNEL_COMPILE_THREADS=4 \
  -DSGL_KERNEL_ENABLE_SM90A=OFF \
  -DSGL_KERNEL_ENABLE_SM100A=OFF \
  -DSGL_KERNEL_ENABLE_FA3=OFF" \
  CMAKE_BUILD_PARALLEL_LEVEL=4 \
  uv pip install --no-build-isolation --no-deps /sglang/sgl-kernel

# ─── runtime ──────────────────────────────────────────────────────────────────
# cudnn-runtime saves ~7 GB vs devel; Triton/flashinfer ship their own backends.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python3 python3-dev ca-certificates libnuma1 && \
  rm -rf /var/lib/apt/lists/*

COPY --link --from=sglang-builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
  PATH="/opt/venv/bin:$PATH"

COPY --link --chmod=755 entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
