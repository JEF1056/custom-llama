# syntax=docker/dockerfile:1.7
# vLLM v0.22.0 inference server.
# Uses the official vLLM OpenAI-compatible image as base.
# Build: docker build -t vllm-server .
# Run:   docker compose up vllm-server

FROM vllm/vllm-openai:v0.22.0

COPY --link --chmod=755 entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
