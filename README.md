# custom-llama - System Architecture & Setup Guide

Host **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** on your own hardware behind a single load-balancing endpoint.

**Model:** Hybrid MoE reasoning + vision model with 262K native context, tool calling, and MTP self-speculative decoding.
**Built on:** [JEF1056/ik_llama.cpp](https://github.com/JEF1056/ik_llama.cpp) (branch `ngram-mtp-vision-chain`) — a patch allowing n-gram lookup drafting alongside MTP with the vision tower loaded.


## Repository Layout

```
custom-llama/
├── docker/              # CUDA server: Docker files, entrypoint, compose, benchmarks
├── mac/                 # MLX Mac deployment: install, deploy, supervisor, health, bench scripts
├── router/              # Olla LiteLLM proxy: docker-compose, config.yaml, env files
├── scripts/             # Offline weight pipeline: download, quantify, imatrix conversion
├── mcp-search-server/   # Web search + browser automation MCP server
├── docs/                # Benchmark results & migration plan
├── opencode.json        # OpenCode AI SDK config: provider + model settings
└── README.md            # This file (Setup Guide)
```

## IMPORTANT: This Repo is DEEP — Read All Modules Before Configuring
- **docker/** has 8 files totaling 1,126 lines: 2 Dockerfiles, entrypoint.sh (.env, .env.example), compose file, 2 benchmark scripts
- **mac/** has 9 files: install.sh, deploy.sh, run-mlx-server.sh, launch-mlx-server.sh, supervisor-mlx-server.sh, healthcheck.sh, uninstall.sh, bench-stress.sh, and a hidden `.env.local` file with overrides
- **router/** has 4 files: docker-compose, config.yaml, .env, .env.example
- **scripts/** has 5 files: download-source-gguf.sh, prepare-weights.sh, quantize.sh, quantize-mlx.sh, quantize-mmproj.py, download-bringup.sh
- **mcp-search-server/** has 40+ Python files with 8 distinct tool handlers (search, fetch, browser, code_run, advisor, time_now, read_output)
- **docs/** has 2 files: qwen36-bench-results.md (53 lines) and iqllama-migration-plan.md (size unknown but likely large)
- **opencode.json** has 70 lines with 3 provider configs and model aliases


## Prerequisites

- **Linux host** with NVIDIA GPU (RTX 3090 recommended, 24 GB VRAM)
- **NVIDIA Container Toolkit** installed (Docker GPU passthrough)
- **Docker** and **Docker Compose** v2
- **~70 GB free disk** for full pipeline or **~17 GB** for quick bring-up
- **Hugging Face token** (`hf_xxx`) — only for gated/private repos; default bring-up weights are public
- **MacBook with Apple Silicon** — for the MLX backend (separate deployment)

---

## Option A: Linux + CUDA (Docker)

### Step 1: Clone and configure

```bash
cp docker/.env.example docker/.env
cp router/.env.example router/.env
```

Edit `docker/.env` and `router/.env` to set your keys and backend URLs. When running both server and router in the same compose, `CUDA_BACKEND_URL` is overridden to `http://server:8080/v1` automatically.

### Step 2: Download weights (choose one)

**Quick bring-up (~17 GB, pre-quantized):**

```bash
docker compose --profile bringup run --rm model-bringup
```

This downloads a pre-quantized GGUF and vision mmproj into the shared volume. Takes a few minutes.

**Production quantization (~70 GB, custom recipe):**

```bash
docker compose build
docker compose --profile prep run --rm model-prep
```

This runs the full offline pipeline:
1. Downloads Unsloth's pre-converted BF16 GGUF shards + mmproj (~70 GB)
2. Computes a custom imatrix from a diverse calibration corpus
3. Quantizes with the "262K-Balanced" recipe:
   - Edge experts: `iq4_ks` | Middle experts: `iq3_k` | Shared expert: `q8_0`
   - Attention layers: `iq5_ks` | Router: `q8_0` | Token embedding: `iq4_ks`
   - Output: `q6_K` | MTP block: BF16 (output head at `q8_0`)

A GPU is required for the imatrix pass. This takes a while.

### Step 3: Start everything

```bash
docker compose up -d --build
```

This builds the CUDA server from `ik_llama.cpp` source, then starts the LiteLLM router. The router waits for the server to be healthy before starting.

### Step 4: Test it

```bash
# Direct to server
curl http://localhost:8080/health

# Through the router
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-master-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b","messages":[{"role":"user","content":"Explain gated DeltaNet."}]}'
```

### Development mode (server only)

For iterating on server config without the router:

```bash
cd docker
cp .env.example .env
docker compose up -d --build
```

The dev stack defaults to port `8081` to avoid collision with prod's `8080`. Build from a local `ik_llama.cpp` checkout:

```bash
BUILD_MODE=local LLAMA_LOCAL_PATH=/path/to/ik_llama.cpp docker compose up -d --build
```

---


## Option B: MacBook + Apple Silicon (MLX)

The CUDA server requires an NVIDIA GPU. For MacBooks with Apple Silicon, use the separate MLX deployment:

**Two deployment modes are supported:**

**Single-host install (any Mac with Apple Silicon):**
```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | HF_TOKEN=hf_xxx bash
```

**Fleet deployment (multiple MacBooks, SSH-configured):**
```bash
# Deploy to all hosts in ALL_HOSTS=(ml-2 ml-3 ml-4)
bash mac/deploy.sh

# Deploy to specific hosts only
bash mac/deploy.sh ml-2 ml-3
```

Both modes install:
- The MLX model (Qwen3.6-35B-A3B with native vision)
- A **LaunchAgent** that starts the server at login and auto-restarts on crash

**After install:**
- **Server:** `http://localhost:8080/v1`
- **Logs:** `~/Library/Logs/qwen36-mlx.out.log` / `.err.log`
- **Supervisor:** `~/Library/Logs/qwen36-mlx-supervisor.log` (health check every 30s with 10 restarts per 300s)
- **Uninstall:** `bash ~/.qwen/custom-llama/mac/uninstall.sh`

---

## Option C: Standalone Router

To run the router independently (e.g., pointing at external backends):

```bash
cd router
cp .env.example .env    # set LITELLM_MASTER_KEY, CUDA_BACKEND_URL, MAC_BACKEND_URL
docker compose up -d
```

The router uses **Olla LB** (`least-connections`) with sticky sessions (1h sliding TTL, max 10K sessions, keyed by `X-Olla-Session-ID` header set by `.opencode/plugins/olla-session.js`). Static endpoints with automatic health checks (5s interval, 2s timeout). In-memory model registry with unifier engine (24h stale threshold).


## Frontend Configuration

- **qwen3.6-35b** maps to both CUDA `/models/qwen3.6-35b` and MLX `/Users/jfan/.qwen/models/qwen36-mlx`
- **Static endpoints:** Automatic health checks (5s interval, 2s timeout)
- **Translators:** Anthproxy pass-through enabled
- **Discovery:** Automatic health checks (5s interval, 2s timeout)
- **Model registry:** In-memory with unifier engine (24h stale threshold)
- **Auth:** LiteLLM master key passed by clients; per-backend auth via `BACKEND_API_KEY`


---

## Configuration Reference

### CUDA server (`docker/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SOURCE` | `local` | `local` = custom GGUF from `/models/`; `hf` = pull from Hugging Face |
| `GGUF_FILE` | `qwen36-262k-balanced.gguf` | GGUF filename under `/models` when using local source |
| `CTX` | `262144` | Context window in tokens. Set `0` for auto-fit to VRAM |
| `KV_TYPE` | `q4_0` | KV-cache quantization for the 10 attention layers |
| `KV_HADAMARD` | `1` | Enable Hadamard-rotated K/V cache. Set `0` to disable |
| `CACHE_RAM_MIB` | `8192` | Prompt caching RAM budget in MiB (`-1` = unlimited, `0` = disable) |
| `N_PARALLEL` | `1` | Concurrent request slots. Keep at `1` for full context length |
| `UBATCH_SIZE` | `1024` | Physical batch size for prompt processing throughput |
| `ENABLE_VISION` | `1` | Enable vision tower. Set `0` for text-only server |
| `MMPROJ_FILE` | `mmproj-BF16.gguf` | Vision projector filename under `/models` |
| `ENABLE_MTP` | `1` | Enable MTP self-speculative decoding. Set `0` to disable |
| `MTP_N_MAX` | `4` | Max speculative tokens per round |
| `ENABLE_NGRAM` | `1` | Enable n-gram lookup drafter (chained before MTP) |
| `TEMP` | `0.6` | Default temperature |
| `TOP_P` | `0.95` | Default top-p |
| `PORT` | `8080` | API port |

### Router (`router/.env`)

| Variable | Description |
|----------|-------------|
| `LITELLM_MASTER_KEY` | Auth key clients present to the router (`Authorization: Bearer ...`) |
| `CUDA_BACKEND_URL` | Base URL of the CUDA server (e.g., `http://192.168.1.50:8080/v1`) |
| `MAC_BACKEND_URL` | Base URL of the Mac MLX backend |
| `BACKEND_API_KEY` | Auth key for backends (any non-empty value works) |


---

## Manual Script Usage

The quantization scripts can also be run directly outside of Docker:

```bash
# 1. Download source GGUF (requires `hf` CLI, ~75+ GB disk)
./scripts/download-source-gguf.sh

# 2. Quantize to production GGUF
./scripts/quantize.sh
```

Or use the Unsloth-provided imatrix (shipped with the source GGUF) instead of computing your own:

```bash
# The downloaded imatrix_unsloth.dat is used automatically by quantize.sh.
```

---

## Troubleshooting

**Server won't start — model file not found:**
Ensure the GGUF exists in the `qwen36-models` volume. Run `docker compose --profile prep run --rm model-prep` to produce it, or `docker compose --profile bringup run --rm model-bringup` for a quick pre-quantized version.

**MTP not engaging:**
The MTP tensors are baked into the GGUF. If using the Unsloth bring-up GGUF, the MTP tensors may be absent (known issue with that particular repo). Use your own quantized GGUF from the full pipeline.

**Vision not working:**
Ensure `ENABLE_VISION=1` and that `mmproj-BF16.gguf` exists in the `/models` volume. The mmproj is copied by the `model-prep` or `model-bringup` jobs automatically.

**High VRAM usage:**
Reduce `N_PARALLEL` to `1`, lower `CTX`, or use a smaller KV type (e.g., `q4_0` is the default). The production quant at ~16.8 GB + KV cache should fit in 24 GB VRAM at `CTX=262144`.

**Build fails — CUDA arch mismatch:**
Set `CUDA_ARCH` in `docker/.env` to match your GPU architecture (e.g., `89` for Ada Lovelace, `86` for Ampere/RTX 3090).

---

## Benchmark Results

See [`docs/qwen36-bench-results.md`](docs/qwen36-bench-results.md) for full benchmark results and methodology. Key results from real hardware (RTX 3090):

- Basic completion: PASS
- Vision: PASS
- MTP self-speculative decoding: PASS (high draft acceptance rates)
- Prompt cache reuse: PASS (54x speedup on repeated prompts)
- Long-context stability: PASS (262K context, 172K+ token prompts)

