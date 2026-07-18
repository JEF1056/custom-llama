# CUDA server for Bonsai-27B (1-bit) on an RTX 3090, built from the TurboQuant+
# llama.cpp fork (Q1_0 1-bit weights + qwen35 arch + TurboQuant KV cache).
#
# Multi-stage: a devel image compiles llama-server for Ampere (sm_86); a slim
# runtime image ships only the binary + shared libs.
#
# DSpark speculative decoding is built in (from the fork's dspark-integration
# branch): the drafter loads as a separate GGUF and runs alongside prompt/prefix
# caching. Base inference has 1-bit weights, the full 262K context and tool
# calling (--jinja).
#
# BUILD_MODE controls where the llama.cpp source comes from:
#   git   (default) - clones LLAMA_REPO at LLAMA_REF
#   local           - copies from ./llama-local/ in the build context
#                     (place or rsync the source there before building)
# CUDA 13.3.0: latest version validated by upstream llama.cpp CI (the `cuda13`
# tag in .github/workflows/docker.yml). Ships CCCL >= 3.2, which enables the
# native cub::DeviceTopK fast path in ggml_top_k (CUDA 12.x fell back to a
# full-vocab argsort). Ampere sm_86 (RTX 3090) is still supported in CUDA 13.
ARG CUDA_VERSION=13.3.0
ARG BUILD_MODE=git

# --- source: git -------------------------------------------------------------
FROM alpine:3.20 AS source-git
ARG LLAMA_REPO=https://github.com/JEF1056/llama-cpp-turboquant.git
ARG LLAMA_REF=dspark-integration
RUN apk add --no-cache git ca-certificates && \
    git clone --depth 1 --branch "${LLAMA_REF}" "${LLAMA_REPO}" /src

# --- source: local -----------------------------------------------------------
# Build context must contain the llama.cpp source at ./llama-local/.
# Example: rsync -a --exclude='.git' /path/to/llama-cpp-turboquant/ docker/llama-local/
FROM alpine:3.20 AS source-local
COPY llama-local/ /src/

# --- select source (BUILD_MODE must be 'git' or 'local') ---------------------
FROM source-${BUILD_MODE} AS source

# --- build stage -------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# RTX 3090 = Ampere sm_86. Override for other cards.
ARG CUDA_ARCH=86

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        build-essential \
        libcurl4-openssl-dev \
        libssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY --from=source /src .

# The CUDA driver (libcuda.so.1) is injected by the NVIDIA Container Toolkit at
# runtime and is absent from the build image, which ships only a stub named
# libcuda.so. Expose it under the versioned soname and put the stubs dir on
# LD_LIBRARY_PATH: ld resolves a shared library's transitive DT_NEEDED (here
# libggml-cuda.so -> libcuda.so.1, for cuMem*/cuDevice*) via LD_LIBRARY_PATH /
# rpath-link, not the -L search path, so LIBRARY_PATH alone is not enough. The
# real driver is used at run time; this stub only satisfies the link.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# CURL on so llama-server can pull GGUF weights via -hf at runtime; OpenSSL on
# so those downloads work over HTTPS (without it the server aborts with "HTTPS
# is not supported"). The embedded Web UI is disabled (LLAMA_BUILD_UI +
# LLAMA_USE_PREBUILT_UI OFF): this is a headless API server, and the prebuilt
# UI-asset download breaks the build (the HF bundle is missing loading.html).
# Server API/tooling is unaffected.
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_CURL=ON \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --target llama-server -j

# --- runtime stage -----------------------------------------------------------
# Must match the build stage's Ubuntu release: the binary links against the
# glibc/libstdc++ (GLIBC_2.38, GLIBCXX_3.4.32) shipped by 24.04, which the
# 22.04 runtime image does not provide.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Binary + all ggml/llama shared objects, kept together.
COPY --from=build /src/build/bin/ /opt/bonsai/bin/
ENV PATH="/opt/bonsai/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/bonsai/bin"

# Fuse the DSpark markov resample into the drafter decode graph (removes the
# per-round draft-logits device->host->device round-trip). On by default in the
# fork; set to 0 to fall back to the host/CUDA resample path.
ENV LLAMA_DSPARK_MARKOV_INGRAPH=1

WORKDIR /workspace
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
