# custom-llama — Setup Guide

Host **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** on your own hardware behind a single load-balancing endpoint.

**Model:** Hybrid MoE reasoning + vision model with 262K native context, tool calling, and MTP self-speculative decoding.

**Built on:** [JEF1056/ik_llama.cpp](https://github.com/JEF1056/ik_llama.cpp) (branch `ngram-mtp-vision-chain`) — a patch allowing n-gram lookup drafting alongside MTP with the vision tower loaded.

---

## Architecture

| Component | Backend | Location |
|-----------|---------|----------|
| `docker/` | `ik_llama.cpp` + CUDA (RTX 3090) | Linux host via Docker |
| `mac/` | MLX (`mlx-vlm`) | MacBook / Apple Silicon |
| `router/` | LiteLLM proxy | Co-located with CUDA or standalone |

---

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

Edit `router/.env` to set your master key and backend URLs. When running both server and router in the same compose, `CUDA_BACKEND_URL` is overridden to `http://server:8080/v1` automatically.

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

A GPU is required for the imatrix step. This takes a while.

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

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | BONSAI_TOKEN=hf_xxx bash
```

Replace `YOURUSER` with your GitHub account and `hf_xxx` with your Hugging Face read token (required — Bonsai-27B repos are private).

This installs:
- The MLX model (ternary 2-bit for vision, or 1-bit text-only)
- A **LaunchAgent** that starts the server at login and auto-restarts on crash

**After install:**
- **Server:** `http://localhost:8081/v1`
- **Logs:** `~/Library/Logs/bonsai-mlx.out.log` / `.err.log`
- **Uninstall:** `bash ~/.bonsai/custom-llama/mac/uninstall.sh`

---

## Option C: Standalone Router

To run the router independently (e.g., pointing at external backends):

```bash
cd router
cp .env.example .env    # set LITELLM_MASTER_KEY, CUDA_BACKEND_URL, MAC_BACKEND_URL
docker compose up -d
```

The router uses **latency-based routing** with automatic retry/failover. Tune backends and strategy in `router/config.yaml`. By default it routes to both the CUDA server and Mac backend, sending each request to the lowest-latency deployment with 3 retries on failure.

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
# 1. Download source GGUF (requires `hf` CLI, ~70 GB disk)
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
