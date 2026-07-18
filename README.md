# custom-llama — Self-host Qwen3.6-35B-A3B on your own hardware

Host **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** — a hybrid MoE reasoning + vision model with 262K native context, tool calling, and MTP self-speculative decoding — on your own hardware behind a single load-balancing endpoint.

## What's in this repo

| Component | Backend | Where it runs |
|-----------|---------|---------------|
| `docker/` | `ik_llama.cpp` + CUDA (RTX 3090) | Linux host via Docker |
| `mac/` | MLX (`mlx-vlm`) | MacBook / Apple Silicon, auto-starts at login |
| `router/` | LiteLLM proxy | Co-located with CUDA or standalone |

Built on stock **[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** (no forks/patches), with every speedup enabled: mixed-precision quant, 4-bit Hadamard-rotated KV cache, Flash Attention, fused MoE, MTP self-speculative decoding, and prompt caching.

See [`docs/iqllama-migration-plan.md`](docs/iqllama-migration-plan.md) for the full design and source-verified feasibility findings.

---

## Prerequisites

- **Linux host** with NVIDIA GPU (RTX 3090 recommended, 24 GB VRAM)
- **NVIDIA Container Toolkit** installed (for Docker GPU passthrough)
- **Docker** and **Docker Compose** v2
- **~70 GB free disk** for model weights (full pipeline) or **~17 GB** (bring-up only)
- **Hugging Face token** (`hf_xxx`) — only needed for gated/private repos; the default bring-up weights are public
- **MacBook with Apple Silicon** — for the MLX backend (separate deployment)

---

## Quick Start — Fastest path to a running server

This gets you a working server in minutes using a pre-quantized GGUF from Hugging Face.

### 1. Clone and configure

```bash
# Copy env files from examples
cp docker/.env.example docker/.env
cp router/.env.example router/.env

# Edit router/.env — set your master key and backend URLs
# (When running both server + router in the same compose, the CUDA_BACKEND_URL
#  is overridden to http://server:8080/v1 automatically — no host IP needed)
```

### 2. Download bring-up weights (one-time, ~17 GB)

```bash
docker compose --profile bringup run --rm model-bringup
```

This downloads a pre-quantized GGUF and the vision mmproj into the shared volume. Takes a few minutes depending on your internet speed.

### 3. Build and start everything

```bash
docker compose up -d --build
```

This builds the CUDA server from `ik_llama.cpp` source, then starts the LiteLLM router. The router waits for the server to be healthy before starting.

### 4. Test it

```bash
# Direct to server
curl http://localhost:8080/health

# Through the router
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-master-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b","messages":[{"role":"user","content":"Explain gated DeltaNet."}]}'
```

---

## Production Setup — Full custom quantization

The bring-up path above uses a public pre-quantized GGUF for quick validation. For production, run the full offline quantization pipeline to produce your own **"262K-Balanced"** GGUF (~16.8 GB) using a custom quantization recipe tuned for this model.

### Step 1: Prepare the environment

```bash
cp docker/.env.example docker/.env
cp router/.env.example router/.env
```

### Step 2: Build the Docker image

```bash
docker compose build
```

This compiles `llama-server`, `llama-quantize`, `llama-imatrix`, and `llama-gguf-split` from `ik_llama.cpp` `main` with CUDA support for your GPU architecture (default: sm_86 for RTX 3090).

### Step 3: Run the full weights pipeline

```bash
docker compose --profile prep run --rm model-prep
```

This runs the complete offline pipeline:
1. Downloads Unsloth's pre-converted BF16 GGUF shards + mmproj from Hugging Face (~70 GB download)
2. Computes a custom imatrix from a diverse calibration corpus (chat/code/reasoning/tool-calling)
3. Quantizes with the "262K-Balanced" recipe:
   - Edge experts: `iq4_ks` (most sensitive layers)
   - Middle experts: `iq3_k` (sparse bulk)
   - Shared expert: `q8_0`
   - Attention layers: `iq5_ks`
   - Router: `q8_0`
   - Token embedding: `iq4_ks`
   - Output: `q6_K`
   - MTP block: kept at BF16 (except output head at `q8_0`)

This step takes a while (large download + imatrix pass + quantize). A GPU is required for the imatrix step.

### Step 4: Start the server

```bash
docker compose up -d
```

The server automatically picks up the quantized GGUF from the shared volume.

---

## Development Setup — CUDA server only

For iterating on the server configuration without the router:

```bash
cd docker
cp .env.example .env
docker compose up -d --build
```

**Building from a local `ik_llama.cpp` checkout:**

```bash
# Rsync your local source into the build context
rsync -a --exclude='.git' /path/to/ik_llama.cpp/ docker/llama-local/

# Build from local source instead of cloning from git
BUILD_MODE=local docker compose up -d --build
```

After the initial build, `.env` and `entrypoint.sh` changes take effect on the next `docker compose up -d` **without a rebuild**. Only Dockerfile or build-arg changes require `--build`.

---

## MacBook Setup — Apple Silicon (MLX)

The CUDA server requires an NVIDIA GPU. For MacBooks with Apple Silicon, use the separate MLX deployment:

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | BONSAI_TOKEN=hf_xxx bash
```

Replace `YOURUSER` with your GitHub account and `hf_xxx` with your Hugging Face read token (required — the Bonsai-27B repos are private).

This installs:
- The MLX model (ternary 2-bit for vision, or 1-bit text-only)
- A **LaunchAgent** that starts the server at login and auto-restarts on crash

**After install:**
- **Server:** `http://localhost:8081/v1`
- **Logs:** `~/Library/Logs/bonsai-mlx.out.log` / `.err.log`
- **Uninstall:** `bash ~/.bonsai/custom-llama/mac/uninstall.sh`

---

## Standalone Router

The router is included in the prod compose file. To run it independently (e.g., pointing at external backends):

```bash
cd router
cp .env.example .env    # set LITELLM_MASTER_KEY, CUDA_BACKEND_URL, MAC_BACKEND_URL
docker compose up -d
```

The router uses **latency-based routing** with automatic retry/failover. Tune backends and strategy in [`router/config.yaml`](router/config.yaml). By default it routes to both the CUDA server and Mac backend, sending each request to the lowest-latency deployment with 3 retries on failure.

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

## Manual script usage

The quantization scripts can also be run directly outside of Docker:

```bash
# 1. Download source GGUF (requires `hf` CLI, ~70 GB disk)
./scripts/download-source-gguf.sh

# 2. Compute custom imatrix (requires a diverse corpus file)
./scripts/compute-imatrix.sh

# 3. Quantize to production GGUF
./scripts/quantize.sh
```

Or use the Unsloth-provided imatrix instead of computing your own:

```bash
SKIP_OWN_IMATRIX=1 ./scripts/quantize.sh
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

## Benchmark results

See [`docs/qwen36-bench-results.md`](docs/qwen36-bench-results.md) for full benchmark results and methodology. Key results from real hardware (RTX 3090):

- Basic completion: PASS
- Vision: PASS
- MTP self-speculative decoding: PASS (high draft acceptance rates)
- Prompt cache reuse: PASS (54x speedup on repeated prompts)
- Long-context stability: PASS (262K context, 172K+ token prompts)
