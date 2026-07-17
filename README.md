# custom-llama — self-hosted Bonsai-27B (1-bit)

Host [`prism-ml/Bonsai-27B`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) —
a true 1-bit (Q1_0, ~3.5 GB) 27B reasoning + vision model — on your own
hardware, with every speedup enabled, behind a single load-balancing endpoint.

Three components:

| Component | Backend | Where it runs |
|-----------|---------|---------------|
| [`docker/`](docker/)   | llama.cpp + CUDA | RTX 3090 (Linux, Docker) |
| [`mac/`](mac/)         | MLX              | MacBook (Apple Silicon), auto-starts on login |
| [`router/`](router/)   | LiteLLM proxy    | Anywhere; distributes across both |

Speedups wired in: **1-bit weights**, a **quantized KV cache**, flash attention,
full GPU offload, and a reasoning-budget cap. **Prompt caching** (KV/prefix
reuse) is active on both machines. The CUDA server is now built from the
TurboQuant+ llama.cpp fork; **DSpark speculative decoding is being ported into
the fork and is not in this image yet**, so there is no speculative-vs-cache
trade-off here — prompt caching is always on.

**Tool calling** (native OpenAI-style `tool_calls`) and the **full 262K context
window** are enabled on both machines:

- CUDA: the server runs with `--jinja` for tool calling; `CTX=262144` gives the
  model's full window (fits in 24 GB thanks to the quantized KV cache). Serving
  is single-slot (no `--parallel`).
- MLX: the 27B emits native `tool_calls`, and mlx-lm keeps an unbounded KV cache
  (full 262K, limited only by unified memory).

> **Heads up — Hugging Face token required.** The Bonsai-27B repo is currently
> private, so you need an HF **read** token (`BONSAI_TOKEN`) for downloads.

---

## 1. RTX 3090 — CUDA server (Docker)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
cd docker
cp .env.example .env
#   edit .env -> set BONSAI_TOKEN (and tune REASONING_BUDGET / CTX)
docker compose up -d --build
```

The build compiles `llama-server` from the TurboQuant+ fork for the RTX 3090
(sm_86); first boot then downloads the 1-bit weights into a Docker volume
(several GB — be patient; `start_period` is generous). Then:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-27b","messages":[{"role":"user","content":"Explain 1-bit quantization."}]}'
```

Toggles live in `docker/.env` — `CTX`, `KV_TYPE` (`q4_0` default, or `turbo4`
for the fork's TurboQuant KV), `CACHE_REUSE`, `REASONING_BUDGET`, `EXTRA_ARGS`,
plus build knobs `LLAMA_REPO` / `LLAMA_REF` / `CUDA_ARCH`.

---

## 2. MacBook (Apple Silicon) — MLX server, auto-start

One command, from your GitHub raw URL:

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | BONSAI_TOKEN=hf_xxx bash
```

This installs the model and a **LaunchAgent** that starts the MLX server at
login and restarts it on crash (`KeepAlive`). Verify:

```bash
launchctl list | grep bonsai
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-27b","messages":[{"role":"user","content":"hi"}]}'
```

- Logs: `~/Library/Logs/bonsai-mlx.out.log` / `.err.log`
- Uninstall: `bash ~/.bonsai/custom-llama/mac/uninstall.sh`

> Replace `YOURUSER` with your GitHub account after you push this repo (also the
> `CUSTOM_LLAMA_REPO` default in [`mac/install.sh`](mac/install.sh)).

---

## 3. Router — one endpoint across both backends

```bash
cd router
cp .env.example .env
#   edit .env -> set CUDA_BACKEND_URL, MAC_BACKEND_URL, LITELLM_MASTER_KEY
docker compose up -d
```

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-27b","messages":[{"role":"user","content":"hi"}]}'
```

Latency-based routing with automatic retry/failover to the healthy backend, plus
background health checks. Tune in [`router/config.yaml`](router/config.yaml).

---

## Notes

- The CUDA image builds `llama-server` from the TurboQuant+ fork
  ([`JEF1056/llama-cpp-turboquant`](https://github.com/JEF1056/llama-cpp-turboquant),
  branch `bonsai`), which already has the `Q1_0` 1-bit kernels, the `qwen35`
  architecture and the TurboQuant KV cache. Point `LLAMA_REPO` / `LLAMA_REF` at
  a different fork/branch to build something else.
- **DSpark speculative decoding is not in this image yet** — it is still being
  ported into the fork. When it lands it will run alongside the prompt cache
  (no trade-off), unlike the vendor demo where the two are mutually exclusive.
- Not included: TLS/public exposure, the Ternary quality variant, vision/Open
  WebUI demos (all available upstream if you want them later).
