# custom-llama

A self-hosted LLM inference server powered by [SGLang](https://github.com/JEF1056/sglang-turboquant) (TurboQuant fork with fused Triton KV cache). Serves AutoRound INT4 safetensors or GGUF models. Exposed publicly via Cloudflare Tunnel with Cloudflare Access authentication.

**Default model:** [Qwen3.6-27B AutoRound INT4](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — 19 GB, MTP heads in BF16, vision support, NEXTN speculative decoding (~90% acceptance on RTX 3090).

---

## Local quick start

No Cloudflare or secrets needed — just the inference server on this machine.

```bash
# 1. Generate .env from defaults (auto-generates SGLANG_API_KEY, MCP_API_KEY)
python sync-env.py

# 2. Build images
docker compose build sglang-server llama-convert mcp-search-server

# 3. Download the default AutoRound INT4 model (~19 GB)
docker compose run --rm llama-convert download qwen3.6-27b-autoround

# 4. .env already defaults to this model — no changes needed
#    SGLANG_MODEL_PATH=/models/qwen3.6-27b-autoround
#    SGLANG_QUANTIZATION=auto-round
#    SGLANG_SPECULATIVE_ALGO=NEXTN

# 5. Start
docker compose up -d sglang-server mcp-search-server
```

Port 8080 is exposed via `docker-compose.override.yml` (gitignored) for local dev. Test with:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
```

Any OpenAI-compatible client (Cursor, Roo Code, opencode, etc.) points at `http://localhost:8080/v1`.

---

## Images

| Image | Dockerfile | Purpose |
|---|---|---|
| `sglang-server` | `Dockerfile` | SGLang CUDA server, built from `JEF1056/sglang-turboquant` |
| `llama-convert` | `Dockerfile.convert` | CPU-only: download, quantize, convert GGUFs |
| `mcp-search-server` | `mcp-search-server/Dockerfile` | Web search MCP tool |

### SGLang server (`Dockerfile`)

Built from source from `JEF1056/sglang-turboquant` — a fork of [sgl-project/sglang](https://github.com/sgl-project/sglang) with **TurboQuant PR #23135** merged in:

- **AutoRound INT4** — `--quantization auto-round` loads pre-quantized safetensors with MTP heads preserved in BF16. Preferred over AWQ on RTX 3090: 19 GB vs AWQ's 21.56 GB which forces `--enforce-eager`.
- **TurboQuant KV cache** — fused Triton kernels read packed 4-bit KV directly during attention (no dequant buffer). 3.88× KV compression, 93–105% of bf16 decode throughput, CUDA graph compatible.
- **GGUF serving** — `--load-format gguf --quantization gguf` with a HuggingFace tokenizer path.
- **NEXTN speculative decoding** — uses the model's own MTP heads (`--speculative-algo NEXTN --speculative-eagle-topk 1`), no separate draft model required.
- **Reasoning parser** — structured `<think>…</think>` extraction for Qwen3 and DeepSeek models.

> **Pre-requisite:** Merge `sgl-project/sglang` PR #23135 into `JEF1056/sglang-turboquant` on GitHub before building.

#### sgl-kernel build modes

| Mode | Command | Time | Notes |
|---|---|---|---|
| **Precompiled** (default) | `docker compose build sglang-server` | ~1 min | cu124 wheel; targets sm80/sm86/sm89/sm90 |
| **From source** | `docker compose build --build-arg SGL_KERNEL_FROM_SOURCE=1 sglang-server` | ~10–20 min | SM86-optimised; strips SM100+/FA3 |

The precompiled [cu124 wheel](https://docs.sglang.io/whl/cu124) includes RTX 3090 (SM86) and runs on CUDA 12.4+ runtimes. Use source build only if you need to pick up unreleased sgl-kernel changes from the fork.

### Convert image (`Dockerfile.convert`)

Plain `ubuntu:22.04`, zero CUDA dependency. Builds `llama-quantize` CPU-only from the TurboQuant llama.cpp fork (OpenBLAS). Also includes the HF→GGUF conversion pipeline (`convert_hf_to_gguf.py`) and `manage_models.py`.

---

## Architecture

```
              ┌──────────────────────────────────┐
              │         Cloudflare Edge           │
              │  chat.jessfan.com                 │
              └──────────────┬────────────────────┘
                             │ Cloudflare Tunnel (outbound)
                             │ Cloudflare Access (auth required)
              ┌──────────────▼────────────────────┐
              │         Host Machine               │
              │  cloudflared → sglang-server:8080 │
              └───────────────┬────────────────────┘
                              │ llama-net (internal)
                    ┌─────────────────────────┐
                    │  sglang-server :8080   │
                    │  mcp-search-server :3100│
                    └─────────────────────────┘
```

| Interface | URL | Auth |
|---|---|---|
| **Local** | `http://localhost:8080/v1` | None (requires `docker-compose.override.yml`) |
| **Public** | `https://chat.jessfan.com/v1` | Cloudflare Access (Google OAuth / Email) |

---

## Step-by-step setup guide

### Step 1: Merge TurboQuant PR into fork

On GitHub, merge [sgl-project/sglang PR #23135](https://github.com/sgl-project/sglang/pull/23135) into `JEF1056/sglang-turboquant`. This is a one-time manual step required before building the server image.

### Step 2: Create Cloudflare Tunnel

1. Go to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/) → **Networks → Tunnels**
2. Click **Create a tunnel** → **Docker** → copy the token
3. Paste into `.env` as `CF_TUNNEL_TOKEN`
4. Add Public Hostname: `chat.jessfan.com` → `http://sglang-server:8080`

### Step 3: Configure `.env`

```bash
python sync-env.py
```

Edit `.env` and set at minimum:

```bash
# Cloudflare Tunnel token
CF_TUNNEL_TOKEN=eyJhIjoi...

# Model — defaults already set for AutoRound (no changes needed after download)
SGLANG_MODEL_PATH=/models/qwen3.6-27b-autoround
SGLANG_QUANTIZATION=auto-round
SGLANG_SERVED_MODEL_NAME=qwen3.6-27b
```

`SGLANG_API_KEY` and `MCP_API_KEY` are auto-generated by `sync-env.py` if empty.

### Step 4: Prepare a model

```bash
# Build the convert image (CPU-only, no GPU needed)
docker compose build llama-convert

# Option A — AutoRound INT4 safetensors (recommended for RTX 3090)
#   19 GB, MTP heads in BF16, vision support, NEXTN spec decoding ~90% acceptance
docker compose run --rm llama-convert download qwen3.6-27b-autoround

# Option B — AutoRound INT4 safetensors, MoE variant (35B active-3B)
docker compose run --rm llama-convert download qwen3.6-35b-a3b-autoround

# Option C — GGUF with MTP head (for TurboQuant KV cache path)
docker compose run --rm llama-convert convert-st qwen3.6-27b --quant IQ4_XS --mtp

# Option D — small GGUF for testing
docker compose run --rm llama-convert download qwen3.5-4b --quant Q4_K_M

# List all available models
docker compose run --rm llama-convert list
```

> **Gated models:** set `HF_TOKEN=your_token` in `.env`
>
> **WSL2 stability:** set `CONVERT_DOWNLOAD_RATE=300M` and `CONVERT_THREADS=4` to prevent vmmem BSODs.

#### Switching to a GGUF model

```bash
# In .env:
SGLANG_MODEL_PATH=/models/qwen3.6-27b-IQ4_XS-mtp.gguf
SGLANG_TOKENIZER_PATH=Qwen/Qwen3.6-27B
SGLANG_QUANTIZATION=          # leave empty — GGUF sets --quantization gguf automatically
SGLANG_SPECULATIVE_ALGO=NEXTN # requires --mtp GGUF build
```

### Step 5: Build and start

```bash
# Build SGLang server (uses precompiled sgl-kernel wheel by default — fast)
docker compose build sglang-server

# To compile sgl-kernel from source instead (SM86-optimised, ~10–20 min):
docker compose build --build-arg SGL_KERNEL_FROM_SOURCE=1 sglang-server

# Start inference server + MCP search tool
docker compose up -d sglang-server mcp-search-server

# With Cloudflare Tunnel
docker compose up -d
```

Check logs:
```bash
docker compose logs -f sglang-server    # allow up to 5 min for large model load
docker compose logs -f cloudflared      # should show "connected"
```

### Step 6: Test

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/models

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SGLANG_API_KEY" \
  -d '{"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Step 7: Connect a client

```python
import openai

client = openai.OpenAI(
    base_url="https://chat.jessfan.com/v1",
    api_key="none",  # Cloudflare Access handles auth
    default_headers={
        "CF-Access-Client-Id": "<your-client-id>",
        "CF-Access-Client-Secret": "<your-client-secret>",
    },
)

response = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_MODEL_PATH` | `/models/qwen3.6-27b-autoround` | Safetensors directory or absolute path to GGUF inside container |
| `SGLANG_QUANTIZATION` | `auto-round` | Quantization format for safetensors (`auto-round`, `gptq`). Empty for GGUF. |
| `SGLANG_TOKENIZER_PATH` | — | HF repo ID or local tokenizer path. Leave empty for safetensors (bundled). Required for GGUF. |
| `SGLANG_SERVED_MODEL_NAME` | `qwen3.6-27b` | Model alias for `/v1/models` |
| `SGLANG_CONTEXT_LENGTH` | `262144` | Max context window in tokens |
| `SGLANG_MEM_FRACTION_STATIC` | `0.90` | GPU VRAM fraction for weights + KV |
| `SGLANG_MAX_RUNNING_REQUESTS` | `3` | Concurrent request slots |
| `SGLANG_TP_SIZE` | `1` | Tensor parallelism (number of GPUs) |
| `SGLANG_KV_CACHE_DTYPE` | — | KV cache quantization. Leave empty for AutoRound safetensors. |
| `SGLANG_REASONING_PARSER` | `qwen3` | Reasoning extraction parser |
| `SGLANG_SPECULATIVE_ALGO` | `NEXTN` | Speculative decoding algorithm (AutoRound models have MTP heads) |
| `SGLANG_SPECULATIVE_EAGLE_TOPK` | `1` | Draft tree width (1 = linear, recommended for NEXTN) |
| `SGLANG_API_KEY` | auto-generated | Bearer token for API auth |

---

## Docker Compose services

| Service | Purpose |
|---|---|
| `sglang-server` | SGLang inference server (port 8080) |
| `cloudflared` | Cloudflare Tunnel — exposes sglang-server publicly |
| `llama-convert` | Model prep tool (download, convert, quantize) — profile: `convert` |
| `mcp-search-server` | Web search MCP tool (port 3100) |

---

## Troubleshooting

- **Model not loading:** Check `docker compose logs sglang-server`. Common causes: `SGLANG_MODEL_PATH` wrong, missing `SGLANG_TOKENIZER_PATH` (GGUF only), insufficient VRAM.
- **AutoRound OOM:** Lower `SGLANG_MEM_FRACTION_STATIC` (try `0.85`) or reduce `SGLANG_CONTEXT_LENGTH`.
- **NEXTN speculative not working:** For AutoRound models, MTP heads are included — ensure `SGLANG_SPECULATIVE_ALGO=NEXTN` is set. For GGUF, confirm the file was built with `--mtp` (via `convert-st --mtp`).
- **DeltaNet / unsupported arch:** Qwen3.6's hybrid architecture may not load in all SGLang versions. Fall back to a standard model (e.g. `qwen3.5-4b-Q4_K_M.gguf`) to verify the stack, then investigate arch support.
- **TurboQuant KV error at startup:** Set `SGLANG_KV_CACHE_DTYPE=auto` and verify the exact dtype string name once PR #23135 is confirmed merged.
- **Cloudflare Tunnel not connecting:** Verify `CF_TUNNEL_TOKEN`. Check `docker compose logs cloudflared`.
- **GPU not detected:** Verify NVIDIA Container Toolkit. Run `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- **WSL2 BSOD during download/quantize:** Set `CONVERT_DOWNLOAD_RATE=300M` and `CONVERT_THREADS=4` in `.env`.
- **sgl-kernel build fails:** Only applies when `SGL_KERNEL_FROM_SOURCE=1`. Ensure the CUDA devel image matches your driver. Check `gcc`/`g++` version (gcc-13 recommended per SGLang docs). Default precompiled wheel avoids this entirely.
