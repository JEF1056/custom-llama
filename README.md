# custom-llama

A self-hosted LLM inference server powered by [SGLang](https://github.com/JEF1056/sglang-turboquant) (TurboQuant fork with fused Triton KV cache). Serves AutoRound INT4 safetensors models. Exposed publicly via Cloudflare Tunnel with Cloudflare Access authentication.

**Default model:** [Qwen3.6-27B AutoRound INT4](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — 19 GB, MTP heads in BF16, vision support, NEXTN speculative decoding (~90% acceptance on RTX 3090).

---

## Local quick start

No Cloudflare or secrets needed — just the inference server on this machine.

```bash
# 1. Generate .env from defaults (auto-generates SGLANG_API_KEY, MCP_API_KEY)
python sync-env.py

# 2. Build images
docker compose build sglang-server model-prep mcp-search-server

# 3. Download the default AutoRound INT4 model (~19 GB)
docker compose run --rm model-prep download qwen3.6-27b-autoround

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
| `model-prep` | `Dockerfile.modelprep` | Lightweight Python image for downloading AutoRound models |
| `mcp-search-server` | `mcp-search-server/Dockerfile` | Web search MCP tool |

### SGLang server (`Dockerfile`)

Built from source from `JEF1056/sglang-turboquant` — a fork of [sgl-project/sglang](https://github.com/sgl-project/sglang) with **TurboQuant PR #23135** merged in:

- **AutoRound INT4** — `--quantization auto-round` loads pre-quantized safetensors with MTP heads preserved in BF16. Fits 19 GB on RTX 3090, leaving ~5 GB for KV cache and CUDA graphs.
- **TurboQuant KV cache** — fused Triton kernels, 3.88× KV compression, CUDA graph compatible.
- **NEXTN speculative decoding** — uses the model's own MTP heads, no separate draft model required.
- **Reasoning parser** — structured `<think>…</think>` extraction for Qwen3 and DeepSeek models.

> **Pre-requisite:** Merge `sgl-project/sglang` PR #23135 into `JEF1056/sglang-turboquant` on GitHub before building.

#### sgl-kernel build modes

| Mode | Command | Time | Notes |
|---|---|---|---|
| **Precompiled** (default) | `docker compose build sglang-server` | ~1 min | cu124 wheel; targets sm80/sm86/sm89/sm90 |
| **From source** | `docker compose build --build-arg SGL_KERNEL_FROM_SOURCE=1 sglang-server` | ~10–20 min | SM86-optimised; strips SM100+/FA3 |

### Model prep image (`Dockerfile.modelprep`)

Lightweight Python 3.12 + `huggingface_hub` + `hf_transfer`. Downloads AutoRound INT4 safetensors into `./models/<name>/` via `snapshot_download`.

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

# Model defaults are already correct after downloading:
SGLANG_MODEL_PATH=/models/qwen3.6-27b-autoround
SGLANG_QUANTIZATION=auto-round
SGLANG_SERVED_MODEL_NAME=qwen3.6-27b
```

`SGLANG_API_KEY` and `MCP_API_KEY` are auto-generated by `sync-env.py` if empty.

### Step 4: Download a model

```bash
# Build the model-prep image (CPU-only, no GPU needed)
docker compose build model-prep

# Download Qwen3.6-27B AutoRound INT4 (~19 GB, MTP+Vision, RTX 3090 recommended)
docker compose run --rm model-prep download qwen3.6-27b-autoround

# Or the MoE variant (35B active-3B, ~21 GB)
docker compose run --rm model-prep download qwen3.6-35b-a3b-autoround

# List all available models
docker compose run --rm model-prep list
```

> **Gated models:** set `HF_TOKEN=your_token` in `.env` before downloading.

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
| `SGLANG_MODEL_PATH` | `/models/qwen3.6-27b-autoround` | Local safetensors directory inside container |
| `SGLANG_QUANTIZATION` | `auto-round` | Quantization format (`auto-round`, `gptq`) |
| `SGLANG_TOKENIZER_PATH` | — | Leave empty — bundled in AutoRound repo |
| `SGLANG_SERVED_MODEL_NAME` | `qwen3.6-27b` | Model alias for `/v1/models` |
| `SGLANG_CONTEXT_LENGTH` | `262144` | Max context window in tokens |
| `SGLANG_MEM_FRACTION_STATIC` | `0.90` | GPU VRAM fraction for weights + KV |
| `SGLANG_MAX_RUNNING_REQUESTS` | `3` | Concurrent request slots |
| `SGLANG_TP_SIZE` | `1` | Tensor parallelism (number of GPUs) |
| `SGLANG_KV_CACHE_DTYPE` | — | KV cache quantization. Leave empty for default. |
| `SGLANG_REASONING_PARSER` | `qwen3` | Reasoning extraction parser |
| `SGLANG_SPECULATIVE_ALGO` | `NEXTN` | Speculative decoding (AutoRound models have MTP heads) |
| `SGLANG_SPECULATIVE_EAGLE_TOPK` | `1` | Draft tree width (1 = linear, recommended for NEXTN) |
| `SGLANG_API_KEY` | auto-generated | Bearer token for API auth |

---

## Docker Compose services

| Service | Purpose |
|---|---|
| `sglang-server` | SGLang inference server (port 8080) |
| `cloudflared` | Cloudflare Tunnel — exposes sglang-server publicly |
| `model-prep` | Download AutoRound models into `./models/` — profile: `convert` |
| `mcp-search-server` | Web search MCP tool (port 3100) |

---

## Troubleshooting

- **Model not loading:** Check `docker compose logs sglang-server`. Common causes: `SGLANG_MODEL_PATH` wrong, `SGLANG_QUANTIZATION` not set to `auto-round`, insufficient VRAM.
- **OOM / loading fails:** Lower `SGLANG_MEM_FRACTION_STATIC` (try `0.85`) or reduce `SGLANG_CONTEXT_LENGTH`.
- **NEXTN speculative not working:** AutoRound models preserve MTP heads in BF16 — ensure `SGLANG_SPECULATIVE_ALGO=NEXTN` is set. If acceptance rate is near 0%, fall back to `NGRAM`.
- **DeltaNet / unsupported arch:** Qwen3.6's hybrid architecture may not load in all SGLang versions. Check the TurboQuant fork's issue tracker for arch compatibility.
- **TurboQuant KV error at startup:** Set `SGLANG_KV_CACHE_DTYPE=auto` and verify PR #23135 is merged.
- **Cloudflare Tunnel not connecting:** Verify `CF_TUNNEL_TOKEN`. Check `docker compose logs cloudflared`.
- **GPU not detected:** Verify NVIDIA Container Toolkit. Run `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- **sgl-kernel build fails:** Only applies when `SGL_KERNEL_FROM_SOURCE=1`. Ensure the CUDA devel image matches your driver. Default precompiled wheel avoids this entirely.
