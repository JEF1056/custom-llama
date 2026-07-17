# CUDA server for Bonsai-27B (1-bit) on an RTX 3090, built from the TurboQuant+
# llama.cpp fork (Q1_0 1-bit weights + qwen35 arch + TurboQuant KV cache).
#
# Multi-stage: a devel image compiles llama-server for Ampere (sm_86); a slim
# runtime image ships only the binary + shared libs. CUDA 12.4 supports the
# RTX 3090.
#
# DSpark speculative decoding is built in (from the fork's dspark-integration
# branch): the drafter loads as a separate GGUF and runs alongside prompt/prefix
# caching. Base inference has 1-bit weights, the full 262K context and tool
# calling (--jinja).
ARG CUDA_VERSION=12.9.0

# --- build stage -------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        libcurl4-openssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Which fork/branch to build (defaults to the user's fork + dspark-integration
# branch, which contains the DSpark drafter + server capture wiring).
ARG LLAMA_REPO=https://github.com/JEF1056/llama-cpp-turboquant.git
ARG LLAMA_REF=dspark-integration
# RTX 3090 = Ampere sm_86. Override for other cards.
ARG CUDA_ARCH=86

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_REF}" "${LLAMA_REPO}" .

# CURL on so llama-server can pull GGUF weights via -hf at runtime.
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --target llama-server -j

# --- runtime stage -----------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

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

WORKDIR /workspace
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
