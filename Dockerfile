# =============================================================================
# SGLang — JEF1056/sglang-turboquant fork (RTX 3090 / Ampere sm_86)
#
# Stages: base → torch_deps → framework → runtime
#
# Stripped vs. official:
#   - No DeepGEMM (Hopper-only FP8 tiles, unsupported on 3090)
#   - No DeepEP (MoE expert parallelism, requires NVLink topology)
#   - No deepep_builder / devtools_builder / gateway_builder stages
#   - No Mooncake, nixl, FlashMLA (distributed / DeepSeek-specific)
#   - Builds from fork instead of local copy or sgl-project/sglang
#
# Uses uv for Python dependency resolution — avoids pip's resolution-too-deep
# errors caused by cache-dit → diffusers transitive dep explosion.
#
# Build:
#   docker compose build sglang-server
#   docker build --target runtime -t sglang-turboquant .
# =============================================================================

ARG CUDA_VERSION=12.8.1

# -----------------------------------------------------------------------
# base: OS + system packages + uv
# -----------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG CUDA_VERSION
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# GKE default lib/bin locations
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Python 3.12 ships in Ubuntu 24.04 main — no deadsnakes PPA needed.
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
      wget software-properties-common \
      python3.12-full python3.12-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && cd /usr/lib/python3/dist-packages/ \
    && ln -s apt_pkg.cpython-312-*-linux-gnu.so apt_pkg.so

RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
      ca-certificates software-properties-common netcat-openbsd kmod unzip \
      curl wget lsof locales git git-lfs \
      build-essential cmake perl patchelf ccache \
      libopenmpi-dev libnuma1 libnuma-dev numactl \
      ffmpeg \
      libibverbs-dev libibverbs1 libibumad3 librdmacm1 \
      libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
      ibverbs-providers infiniband-diags perftest \
      libssl-dev ninja-build \
      libnccl2 libnccl-dev \
      pybind11-dev \
      protobuf-compiler \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install uv — faster resolver, no resolution-too-deep issues.
# uv pip install --system bypasses virtual-env detection in Docker.
RUN python3 -m pip install uv

# -----------------------------------------------------------------------
# torch_deps: Install sgl-kernel wheel + all sglang Python dependencies.
#   Strategy mirrors a local machine build:
#     1. Pre-install torch from the CUDA-specific PyTorch index so uv
#        treats it as already satisfied when resolving sglang's extras.
#        This sidesteps the fork's [tool.uv.sources] ROCm torch override
#        (only applies to `uv sync`, not `uv pip install`).
#     2. Clone the fork for the dep spec, create a stub sglang package,
#        and install all deps — torch is already present, uv skips it.
#     3. Capture constraints.txt for the framework stage.
# -----------------------------------------------------------------------
FROM base AS torch_deps

ARG CUDA_VERSION
ARG BUILD_TYPE=all
ARG SGL_KERNEL_VERSION=0.4.3
ARG FORK_REPO=https://github.com/JEF1056/sglang-turboquant.git
ARG FORK_BRANCH=feature/carlos-turboquant

WORKDIR /sgl-workspace

# Rust toolchain for the sglang-grpc extension (edition 2024, needs >= 1.85).
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

# Bootstrap: upgrade pip/setuptools/wheel basics, then install sgl-kernel wheel.
# sgl-kernel is a pre-built CUDA-optimised attention/rope wheel; the fork does
# not modify it — all custom kernels are JIT-compiled at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade pip setuptools wheel html5lib six \
    && case $CUDA_VERSION in \
      12.6.1) CUINDEX=126 ;; \
      12.8.1|12.9.1) CUINDEX=129 ;; \
      13.0.1) CUINDEX=130 ;; \
      *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && if [ "$CUDA_VERSION" = "12.6.1" ]; then \
      uv pip install --system --no-deps --reinstall \
        "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_$(uname -m).whl"; \
    elif [ "$CUDA_VERSION" = "12.8.1" ] || [ "$CUDA_VERSION" = "12.9.1" ]; then \
      uv pip install --system --no-deps --reinstall \
        "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu129-cp310-abi3-manylinux2014_$(uname -m).whl"; \
    elif [ "$CUDA_VERSION" = "13.0.1" ]; then \
      uv pip install --system --no-deps --reinstall \
        "sglang-kernel==${SGL_KERNEL_VERSION}"; \
    fi

# Pre-install torch/torchvision/torchaudio from the CUDA-specific PyTorch index.
# Installing torch BEFORE the sglang extras install means uv finds torch already
# satisfied in the environment and skips re-resolving it — no ROCm confusion,
# no version backtracking across thousands of torch candidates.
RUN --mount=type=cache,target=/root/.cache/uv \
    case $CUDA_VERSION in \
      12.6.1) CUINDEX=126 ;; \
      12.8.1|12.9.1) CUINDEX=129 ;; \
      13.0.1) CUINDEX=130 ;; \
      *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && uv pip install --system \
        --index-url "https://download.pytorch.org/whl/cu${CUINDEX}" \
        torch torchvision torchaudio

# Clone fork to get the dep spec (pyproject.toml + Rust crate + proto).
# Shallow clone preserves workspace-relative paths that build.rs / tonic_build expect:
#   /tmp/sglang_deps/
#     python/pyproject.toml
#     rust/sglang-grpc/
#     proto/
RUN git clone --depth=1 --branch "${FORK_BRANCH}" "${FORK_REPO}" /tmp/sglang_deps

# Install all sglang dependencies using a stub sglang package so the real
# source can be installed as an editable package in the next stage without
# re-downloading torch/transformers/etc.
#
# --index-strategy unsafe-best-match: mirrors [tool.uv] in pyproject.toml;
#   picks the best (latest) package across all configured indexes.
# torch is already installed above so uv skips it — no resolver explosion.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    case $CUDA_VERSION in \
      12.6.1) CUINDEX=126 ;; \
      12.8.1|12.9.1) CUINDEX=129 ;; \
      13.0.1) CUINDEX=130 ;; \
      *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && cd /tmp/sglang_deps/python \
    && rm -rf sglang && mkdir -p sglang \
    && touch sglang/__init__.py \
    && touch README.md LICENSE \
    && uv pip install --system \
        --index-strategy unsafe-best-match \
        --extra-index-url "https://download.pytorch.org/whl/cu${CUINDEX}" \
        ".[${BUILD_TYPE}]" \
    && cd /sgl-workspace \
    && rm -rf /tmp/sglang_deps \
    && pip freeze | grep -v '^sglang==' > /sgl-workspace/constraints.txt

# -----------------------------------------------------------------------
# framework: Clone fork + editable install on top of torch_deps
# -----------------------------------------------------------------------
FROM torch_deps AS framework

ARG CUDA_VERSION
ARG BUILD_TYPE=all
ARG FORK_REPO=https://github.com/JEF1056/sglang-turboquant.git
ARG FORK_BRANCH=feature/carlos-turboquant

WORKDIR /sgl-workspace

# sm_86 for RTX 3090; used by any ahead-of-time Torch CUDA extension compile
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Minimal extras needed at runtime / dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -c /sgl-workspace/constraints.txt \
      pytest wheel py-spy

# Clone full fork for editable install (includes Rust source in rust/).
RUN git clone --depth=1 --branch "${FORK_BRANCH}" "${FORK_REPO}" /sgl-workspace/sglang

# Editable install with Rust compilation.
# setuptools-rust must be present BEFORE --no-build-isolation so the build
# backend can import it without an isolated env.
# All sglang Python deps are already installed (torch_deps stage), so uv
# only builds the Rust extension and links the editable package.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    uv pip install --system setuptools-rust \
    && cd /sgl-workspace/sglang \
    && uv pip install --system --no-build-isolation -e "python[${BUILD_TYPE}]" \
    && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
    && find /usr/local/lib/python3.12/dist-packages -type d -name __pycache__ \
         -exec rm -rf {} + 2>/dev/null || true

WORKDIR /sgl-workspace/sglang

# -----------------------------------------------------------------------
# runtime: Production image — CUDA devel base for JIT (Triton, FlashInfer,
#   TurboQuant tq_decode JIT kernels).  No dev tools.
# -----------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime

ARG CUDA_VERSION
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Full CUDA compiler path required by Triton / FlashInfer JIT at runtime
ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
    TORCH_CUDA_ARCH_LIST="8.6"

RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
      python3.12-full python3.12-dev wget \
      ca-certificates netcat-openbsd curl git \
      libopenmpi3 libnuma1 \
      libibverbs1 libibumad3 librdmacm1 libnl-3-200 libnl-route-3-200 \
      ibverbs-providers \
      libssl3 \
      rdma-core infiniband-diags perftest \
      ninja-build \
      libnccl2 libnccl-dev \
      linux-libc-dev \
      libunwind8 \
      libgoogle-glog0v6t64 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Fix Trivy CVEs
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt \
    apt-get update && apt-get install -y --only-upgrade \
      binutils binutils-common binutils-x86-64-linux-gnu libbinutils \
      libctf0 libctf-nobfd0 libgprofng0 libsframe1 \
      libgnutls30t64 \
      libpam-modules libpam-modules-bin libpam-runtime libpam0g \
      libsqlite3-0 libtasn1-6 \
      dpkg dpkg-dev libdpkg-perl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python site-packages from framework (already cleaned of __pycache__)
COPY --from=framework /usr/local/lib/python3.12/dist-packages \
                      /usr/local/lib/python3.12/dist-packages

# Copy SGLang workspace (fork source + editable install metadata)
COPY --from=framework /sgl-workspace /sgl-workspace

# Copy JIT kernel caches
COPY --from=framework /root/.cache/huggingface /root/.cache/huggingface
COPY --from=framework /root/.cache/sglang /root/.cache/sglang

# Copy py-spy profiler
COPY --from=framework /usr/local/bin/py-spy /usr/local/bin/py-spy

WORKDIR /sgl-workspace/sglang

ENV MODEL_DIR=/models
RUN mkdir -p /models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && sed -i 's/\r$//' /entrypoint.sh \
    && python3 -c "import sglang; print('SGLang', sglang.__version__, 'OK')"

# Build provenance
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang-turboquant:dev
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang-turboquant:dev}

LABEL org.opencontainers.image.source="https://github.com/JEF1056/sglang-turboquant" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=300s \
    CMD curl -f "http://localhost:${SGLANG_PORT:-8080}/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
