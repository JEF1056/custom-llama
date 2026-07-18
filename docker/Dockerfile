# CUDA server for Qwen3.6-35B-A3B on an RTX 3090, built from stock
# ikawrakow/ik_llama.cpp (no forks/patches — see docs/iqllama-migration-plan.md).
#
# Multi-stage: a devel image compiles llama-server/llama-quantize/llama-imatrix
# for Ampere (sm_86); a slim runtime image ships only the server binary + libs.
#
# ik_llama.cpp natively supports (verified against source, see the migration
# plan's "Verified findings"): the qwen35moe/qwen3next hybrid (DeltaNet + full
# attention + MoE) architecture, Hadamard-rotated 4-bit KV cache on the
# attention layers (-khad/-vhad), MTP self-speculative decoding
# (--spec-type mtp:...), and Qwen3-VL-family vision (mmproj). The GGUF weights
# themselves are produced OFFLINE by scripts/ (Phase 2 of the migration plan,
# from Unsloth's pre-converted BF16 GGUF) and mounted in via the /models
# volume — this image only builds and runs the engine.
#
# BUILD_MODE controls where the ik_llama.cpp source comes from:
#   git   (default) - clones LLAMA_REPO at LLAMA_REF
#   local           - copies from ./llama-local/ in the build context
#                     (place or rsync the source there before building)
ARG CUDA_VERSION=12.6.3
ARG BUILD_MODE=git

# --- source: git -------------------------------------------------------------
FROM alpine:3.20 AS source-git
ARG LLAMA_REPO=https://github.com/ikawrakow/ik_llama.cpp.git
ARG LLAMA_REF=main
RUN apk add --no-cache git ca-certificates && \
    git clone --depth 1 --branch "${LLAMA_REF}" "${LLAMA_REPO}" /src

# --- source: local -----------------------------------------------------------
# Build context must contain the ik_llama.cpp source at ./llama-local/.
# Example: rsync -a --exclude='.git' /path/to/ik_llama.cpp/ docker/llama-local/
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

# CURL+OpenSSL on so llama-server can pull GGUFs via -hf at runtime (used for
# bring-up against Unsloth's public quant before our own recipe is ready).
# GGML_CUDA_F16 on for faster fp16 CUDA paths on Ampere. Also builds
# llama-quantize/llama-imatrix/llama-gguf-split, used by scripts/ for the
# offline weights pipeline (run inside this same image, see docs/
# iqllama-migration-plan.md Phase 2).
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DGGML_CUDA_F16=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --target llama-server llama-quantize llama-imatrix llama-gguf-split llama-bench -j

# CMake scatters the shared libs (libggml*.so, libllama.so, libmtmd.so, ...)
# across their respective build subdirectories rather than build/bin/; collect
# them alongside the binaries so a single COPY in the runtime stage picks up
# every transitive dependency.
RUN find build -name '*.so*' -exec cp -uP -t build/bin/ {} +

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
        python3 \
        python3-pip \
    && pip3 install --no-cache-dir --break-system-packages 'huggingface_hub[cli]' \
    && rm -rf /var/lib/apt/lists/*

# Server binary + quantize/imatrix tools (used by scripts/ for the offline
# weights pipeline) + all ggml/llama shared objects, kept together.
COPY --from=build /src/build/bin/ /opt/iqllama/bin/
ENV PATH="/opt/iqllama/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/iqllama/bin"

WORKDIR /workspace
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
