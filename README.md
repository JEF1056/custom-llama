# custom-llama — self-hosted Bonsai-27B (1-bit)

Host [`prism-ml/Bonsai-27B`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) —
a true 1-bit (Q1_0, ~3.5 GB) 27B reasoning + vision model — on your own
hardware, with every speedup enabled, behind a single load-balancing endpoint.

Three components:

| Component | Backend | Where it runs |
|-----------|---------|---------------|
| [`docker/`](docker/) | llama.cpp + CUDA | RTX 3090 (Linux, Docker) |
| [`mac/`](mac/) | MLX | MacBook (Apple Silicon), auto-starts on login |
| [`router/`](router/) | LiteLLM proxy | Co-located with CUDA server or standalone |

Speedups wired in: **1-bit weights**, a **quantized KV cache**, flash attention,
full GPU offload, and a reasoning-budget cap. Prompt caching is active: Bonsai
is a hybrid (GDN + attention) arch, so llama.cpp auto-disables `--cache-reuse`
KV-shifting; instead the server's context-checkpoint + prompt-state cache is
used (full sequence-state save/restore). **DSpark speculative decoding** is
integrated in the fork and active by default (`ENABLE_DSPARK=1`) — it rides the
same checkpoint machinery as prompt caching, so both run together with no
trade-off.

**Vision** is enabled on CUDA: Bonsai's vision tower ships as a separate mmproj
GGUF loaded through the Qwen3-VL projector. Set `ENABLE_VISION=0` for a leaner
text-only server.

**Tool calling** (`tool_calls`) and the **full 262K context window** are enabled
on both backends. CUDA uses `--jinja`; MLX keeps an unbounded KV cache.

> **Hugging Face token.** The prism-ml GGUF repos are public — `BONSAI_TOKEN`
> can be left empty. Set it only for private/gated repos.

---

## Compose files

| File | Purpose |
|------|---------|
| [`docker-compose.yml`](docker-compose.yml) | **Prod**: builds CUDA server from git + starts router together |
| [`docker/docker-compose.yml`](docker/docker-compose.yml) | **Dev**: CUDA server only, supports local source builds |

---

## 1. Prod — CUDA server + router (one command)

```bash
cp docker/.env.example docker/.env   # set BONSAI_TOKEN, tune CTX / KV_TYPE etc.
cp router/.env.example router/.env   # set LITELLM_MASTER_KEY, MAC_BACKEND_URL
docker compose up -d --build
```

Builds `llama-server` from the TurboQuant+ fork, downloads weights on first
boot, and starts the LiteLLM router on port 4000. The router reaches the CUDA
server over an internal Docker network — no host IP needed in `router/.env`.

```bash
# direct server
curl http://localhost:8080/health

# through router
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-27b","messages":[{"role":"user","content":"Explain 1-bit quantization."}]}'
```

---

## 2. Dev — CUDA server only (docker/docker-compose.yml)

```bash
cd docker
cp .env.example .env
docker compose up -d --build
```

### Building from a local fork

Set `BUILD_MODE=local` and provide the source under `docker/llama-local/`:

```bash
rsync -a --exclude='.git' /path/to/llama-cpp-turboquant/ docker/llama-local/
BUILD_MODE=local docker compose up -d --build
```

After the initial build, `.env` and `entrypoint.sh` changes take effect on the
next `docker compose up -d` with **no rebuild**. Only Dockerfile or build-arg
changes require `--build`.

Available knobs in `docker/.env`: `CTX`, `KV_TYPE` (`q4_0` default, or `turbo4`
for the fork's TurboQuant KV), `CACHE_RAM_MIB`, `REASONING_BUDGET`,
`ENABLE_VISION` / `MMPROJ_FILE`, `ENABLE_DSPARK` / `DSPARK_DRAFT_FILE`,
`EXTRA_ARGS`. Build knobs: `LLAMA_REPO` / `LLAMA_REF` / `CUDA_ARCH`.

---

## 3. MacBook (Apple Silicon) — MLX server, auto-start

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | BONSAI_TOKEN=hf_xxx bash
```

Installs a **LaunchAgent** that starts the MLX server at login and restarts on
crash. Verify:

```bash
launchctl list | grep bonsai
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-27b","messages":[{"role":"user","content":"hi"}]}'
```

- Logs: `~/Library/Logs/bonsai-mlx.out.log` / `.err.log`
- Uninstall: `bash ~/.bonsai/custom-llama/mac/uninstall.sh`
- Vision opt-in: `ENABLE_VISION=1 ... | bash`. On Apple Silicon this serves
  the ternary (2-bit) MLX build (~7 GB); the 1-bit build is text-only for now.

> Replace `YOURUSER` with your GitHub account (also in `mac/install.sh`).

---

## 4. Router — standalone

The router is included in the prod `docker-compose.yml`. To run it independently:

```bash
cd router
cp .env.example .env   # set LITELLM_MASTER_KEY, CUDA_BACKEND_URL, MAC_BACKEND_URL
docker compose up -d
```

Latency-based routing with automatic retry/failover. Tune backends and routing
strategy in [`router/config.yaml`](router/config.yaml).

---

## Notes

- The CUDA image is built from the TurboQuant+ fork
  ([`JEF1056/llama-cpp-turboquant`](https://github.com/JEF1056/llama-cpp-turboquant),
  branch `dspark-integration`), which has Q1_0 kernels, the qwen35 architecture,
  TurboQuant KV cache, and DSpark speculative decoding. Point `LLAMA_REPO` /
  `LLAMA_REF` at a different fork/branch to build something else.
- **DSpark** is active by default (`ENABLE_DSPARK=1`). Set `ENABLE_DSPARK=0` for
  a plain server. The drafter is a separate ~1.8 GB GGUF downloaded on first
  boot alongside the main weights.
- **Vision** is enabled on CUDA (`ENABLE_VISION=0` to disable). On Mac it is
  opt-in at install time.
- The prod compose exposes the CUDA server on `LLAMA_PORT` (default 8080) and
  the router on `ROUTER_PORT` (default 4000). Set these in the shell or a root
  `.env` to change ports without editing the compose file.

