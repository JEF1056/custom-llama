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
#   local           - copies from an external host directory via a named
#                     build context (see docker-compose.yml's
#                     `additional_contexts: llama-src=...`) - no in-repo
#                     copy needed. Default host path: ~/ik_llama.cpp
#                     (our JEF1056/ik_llama.cpp fork with the ngram+MTP+
#                     vision patch - see docs/iqllama-migration-plan.md).
ARG CUDA_VERSION=13.3.0
ARG BUILD_MODE=git

# --- source: git -------------------------------------------------------------
FROM alpine:3.20 AS source-git
ARG LLAMA_REPO=https://github.com/ikawrakow/ik_llama.cpp.git
ARG LLAMA_REF=main
RUN apk add --no-cache git ca-certificates && \
    git clone --depth 1 --branch "${LLAMA_REF}" "${LLAMA_REPO}" /src

# --- source: local -------------------------------------------------------------
# Sourced from the "llama-src" named build context (an external host
# directory, e.g. ~/ik_llama.cpp), NOT a path inside this build context.
# See docker-compose.yml's `build.additional_contexts.llama-src`.
FROM alpine:3.20 AS source-local
COPY --from=llama-src . /src/

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
ARG BUILD_JOBS=4
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DGGML_CUDA_F16=ON \
        -DGGML_CUDA_NO_VMM=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --target llama-server llama-quantize llama-imatrix llama-gguf-split llama-bench -j ${BUILD_JOBS}

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
    && pip3 install --no-cache-dir --break-system-packages 'huggingface_hub[cli]' numpy \
    && rm -rf /var/lib/apt/lists/*

# Server binary + quantize/imatrix tools (used by scripts/ for the offline
# weights pipeline) + all ggml/llama shared objects, kept together.
COPY --from=build /src/build/bin/ /opt/iqllama/bin/
# GGUF imatrix converter — needed to convert Unsloth's GGUF-format imatrix to
# the binary DAT format expected by ik_llama.cpp's llama-quantize --imatrix.
COPY --from=build /src/convert_imatrix_gguf_to_dat.py /opt/iqllama/
# Fix 1: counts[0] on a 0-d array raises TypeError; flat[0] works for any shape.
# Fix 2: len(entry.values) returns only the first dim for MoE tensors; .size gives total elements.
RUN sed -i \
    -e 's/entry\.counts\[0\]/entry.counts.flat[0]/' \
    -e 's/len(entry\.values)/entry.values.size/' \
    /opt/iqllama/convert_imatrix_gguf_to_dat.py
COPY --from=build /src/gguf-py/ /opt/iqllama/gguf-py/
RUN printf "%s\n" \
    "filepath = '/opt/iqllama/gguf-py/gguf/constants.py'" \
    "with open(filepath, \"r\") as f: lines = f.readlines()" \
    "in_enum = False; seen_keys = set(); new_lines = []" \
    "for line in lines:" \
    "    if line.startswith(\"class MODEL_TENSOR\"): in_enum = True" \
    "    elif in_enum and line.startswith(\"class \"): in_enum = False" \
    "    elif in_enum and not line.startswith(\" \") and line.strip() and not line.strip().startswith(\"#\"): in_enum = False" \
    "    if in_enum:" \
    "        stripped = line.strip()" \
    "        if stripped and not stripped.startswith(\"#\"):" \
    "            parts = stripped.split(\"=\")" \
    "            if len(parts) >= 2 and \"auto()\" in parts[1]:" \
    "                key = parts[0].strip()" \
    "                if key in seen_keys: continue" \
    "                seen_keys.add(key)" \
    "    new_lines.append(line)" \
    "with open(filepath, \"w\") as f: f.writelines(new_lines)" \
    > /tmp/patch_constants.py && python3 /tmp/patch_constants.py && rm /tmp/patch_constants.py
ENV PATH="/opt/iqllama/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/iqllama/bin"

WORKDIR /workspace
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
