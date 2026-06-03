# syntax=docker/dockerfile:1.7
# vLLM inference server.
#
# Official image (default):
#   docker compose build vllm-server
#
# Fork overlay (Python/Triton changes only, ~30s build):
#   VLLM_FORK_REPO=https://github.com/you/vllm.git \
#   VLLM_FORK_BRANCH=feat/k4v2 \
#   docker compose build vllm-server
#
# The fork overlay preserves compiled C/CUDA extensions from the base image
# and only replaces Python source (including Triton kernels, which are
# JIT-compiled at runtime). For C++/CUDA kernel changes, use vLLM's
# upstream docker/Dockerfile instead (30-60 min full build).

ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.22.0
FROM ${VLLM_BASE_IMAGE}

ARG VLLM_FORK_REPO=https://github.com/JEF1056/vllm-turboquant.git
ARG VLLM_FORK_BRANCH=turboquant-k4v2-nc

# If VLLM_FORK_REPO is set, clone and overlay Python source on the installed
# package. Compiled .so extensions from the base image stay untouched.
# If empty, this layer is a no-op and the official image is used as-is.
RUN if [ -n "${VLLM_FORK_REPO}" ]; then \
      echo "Building from fork: ${VLLM_FORK_REPO}@${VLLM_FORK_BRANCH}" \
      && apt-get update -qq && apt-get install -y -qq git > /dev/null && rm -rf /var/lib/apt/lists/* \
      && git clone --depth 1 --branch "${VLLM_FORK_BRANCH}" "${VLLM_FORK_REPO}" /tmp/vllm-fork \
      && VLLM_SITE=$(python3 -c "import vllm; import pathlib; print(pathlib.Path(vllm.__path__[0]).parent)") \
      && cp -a /tmp/vllm-fork/vllm "${VLLM_SITE}/vllm" \
      && rm -rf /tmp/vllm-fork \
      && python3 -c "import vllm; print(f'vLLM {vllm.__version__} (fork overlay)')"; \
    else \
      echo "Using official vLLM image"; \
    fi

COPY --link --chmod=755 entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
