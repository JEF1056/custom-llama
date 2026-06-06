# custom-llama

A self-hosted LLM inference server built around [llama.cpp (TurboQuant + MTP fork)](https://github.com/JEF1056/llama-cpp-turboquant/tree/llama-next). Exposed publicly via Cloudflare Tunnel with Cloudflare Access authentication.

**Default model:** [qwopus3.6-27B-v2](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-GGUF) — a reasoning model with native MTP speculative decoding support.

**35B model:** [qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — hybrid MoE (DeltaNet + MoE Attention), 3.8B active params. Also available as [APEX MTP mixed-precision quants](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF) (`qwen3.6-35b-a3b-apex`, default: `APEX-MTP-I-Compact`).

---

## Local quick start

No Cloudflare or secrets needed — just the inference server on this machine.

```bash
python sync-env.py
docker compose build llama-server llama-convert mcp-search-server
docker compose run --rm llama-convert convert-st qwopus3.6-27b --quant IQ4_XS --mtp
docker compose up -d llama-server mcp-search-server
```

Port 8080 is not exposed by default. Create `docker-compose.override.yml` (gitignored) to open it on localhost:

```yaml
services:
  llama-server:
    ports:
      - "8080:8080"
    networks:
      - llama-net
      - host-bridge

networks:
  host-bridge:
    driver: bridge
```

Then `docker compose up -d llama-server` (not `restart` — that won't re-read the config). Test with `curl http://localhost:8080/health`.

Any OpenAI-compatible client (Cursor, Roo Code, LM Studio, etc.) points at `http://localhost:8080/v1`.

> **Without MTP:** if you want a faster first run (skip the safetensors download), use the prebuilt GGUF instead.
> Comment out `LLAMA_MODEL` and `LLAMA_SPEC_TYPE` in `.env`, then:
> `docker compose run --rm llama-convert download qwopus3.6-27b --quant IQ4_XS`

---

## Models

### qwopus3.6-27B-v2 (default, loads on first request)

| Property     | Value                                                  |
| ------------ | ------------------------------------------------------ |
| Model        | qwopus3.6-27B-v2                                       |
| Quant        | IQ4_XS + MTP (~15 GB)                                  |
| Architecture | Dense transformer, 64 GQA attention layers             |
| Context      | 150K (native 32K; extended via RoPE scaling)           |
| Capabilities | Reasoning, tool use, MTP speculative decoding          |
| MTP speedup  | ~2–2.5× tok/s vs. baseline (requires MTP-capable GGUF) |

**VRAM budget (RTX 3090, 24 GB):**

| Component                   | Size                            |
| --------------------------- | ------------------------------- |
| Model (IQ4_XS)              | ~15.0 GB                        |
| KV cache (turbo3, 150K ctx) | ~2.5 GB                         |
| draft-mtp KV cache          | ~0.3 GB                         |
| CUDA context + compute      | ~1.6 GB                         |
| **Total**                   | **~19.4 GB** (~4.6 GB headroom) |

### qwen3.6-35B-A3B (loads at startup)

| Property      | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| Model         | qwen3.6-35B-A3B                                                       |
| Quant         | IQ4_XS + MTP (~14.5 GB) or APEX-MTP-I-Compact (~17.3 GB)                  |
| Architecture  | Hybrid MoE — 30 DeltaNet (linear-recurrent) + 10 MoE Attention layers |
| Active params | 3.8B of 35B (MoE FFN)                                                 |
| Context       | 128K                                                                  |
| Capabilities  | Reasoning, tool use, MTP speculative decoding                         |

**VRAM budget (RTX 3090, 24 GB) — IQ4_XS:**

| Component                    | Size                            |
| ---------------------------- | ------------------------------- |
| Model (IQ4_XS)               | ~14.5 GB                        |
| DeltaNet recurrent state     | ~1.5 GB                         |
| KV cache (turbo4/2, 65K ctx) | ~0.6 GB                         |
| draft-mtp KV cache           | ~0.1 GB                         |
| compute scratch + CUDA       | ~1.5 GB                         |
| **Total**                    | **~18.2 GB** (~5.8 GB headroom) |

**VRAM budget (RTX 3090, 24 GB) — APEX-MTP-I-Compact:**

| Component                        | Size                            |
| -------------------------------- | ------------------------------- |
| Model (APEX-MTP-I-Compact)       | ~17.3 GB                        |
| DeltaNet recurrent state     | ~1.5 GB                         |
| KV cache (turbo4/2, 65K ctx) | ~0.6 GB                         |
| compute scratch + CUDA       | ~1.5 GB                         |
| **Total**                    | **~20.9 GB** (~3.1 GB headroom) |

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
              │  cloudflared → llama-server :8080  │
              └───────────────┬────────────────────┘
                                │ llama-net (internal)
                    ┌─────────────────────┐
                    │ llama-server :8080  │
                    │  │                   │
                    │  └─ mcp-search-server :3100
                    └─────────────────────┘
```

> **Note:** `mcp-search-server` provides semantic web search with browser automation via MCP. Accessible at `http://mcp-search-server:3100` on the internal `llama-net` network.

| Interface                   | URL / Command                 | Auth                                          |
| --------------------------- | ----------------------------- | --------------------------------------------- |
| **Local**                   | `http://localhost:8080/v1`    | None (requires `docker-compose.override.yml`) |
| **API (Cloudflare Access)** | `https://chat.jessfan.com/v1` | Google OAuth / Email (see below)              |

---

## Step-by-step setup guide

### Step 1: Create Cloudflare Tunnel

1. Go to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/)
2. Navigate to **Networks → Tunnels**
3. Click **Create a tunnel**
4. Select **Docker** as the platform
5. Copy the generated token (looks like `eyJhIjoi...`)
6. Paste it into your `.env` file as `CF_TUNNEL_TOKEN`
7. Add a Public Hostname:
   - **Subdomain**: `api` (or your preferred subdomain)
   - **Domain**: `jessfan.com` (your Cloudflare domain)
   - **Service**: `http://llama-server:8080`
8. Save the tunnel

### Step 2: Set up Cloudflare Access

1. Go to **Zero Trust → Access → Applications**
2. Click **Add an Application**
3. Choose **Add a Cloud Access Application**
4. Enter the same domain you set in the tunnel (e.g., `api.jessfan.com`)
5. Choose an authentication method:
   - **Google OAuth** — uses Google credentials (recommended)
   - **Email/Password** — users get a one-time code via email
6. Set **Who can access** to your email or "Anyone with the domain"
7. Save the application

### Step 3: Configure your `.env` file

```bash
python sync-env.py
```

Edit `.env` and set at minimum:

```bash
# Cloudflare Tunnel token (from Step 1)
CF_TUNNEL_TOKEN=eyJhIjoi...

# Cloudflare Access hostname
CF_ACCESS_HOSTNAME=api.jessfan.com

# Google OAuth credentials for Cloudflare Access
CF_ACCESS_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
CF_ACCESS_GOOGLE_CLIENT_SECRET=your-client-secret

# Optional: API key for llama-server (extra layer of protection)
LLAMA_API_KEY=$(openssl rand -hex 32)
```

### Step 4: Build and prepare models

```bash
# Build the containers
docker compose build
docker compose build llama-convert

# ── 27B (default server model) ─────────────────────────────────────────────

# Option A (recommended): MTP-capable GGUF from safetensors — ~2–2.5× faster generation
# Downloads safetensors, converts to fp16 GGUF, quantizes, cleans up.
# TriAttention calibration runs automatically (TRIATTENTION_INPUT set in docker-compose.yml).
docker compose run --rm llama-convert convert-st qwopus3.6-27b --quant IQ4_XS --mtp --keep-intermediate
# Output: ./models/qwopus3.6-27b-IQ4_XS-mtp.gguf  +  ./models/qwopus3.6-27b-triattention.bin
# .env.default already points LLAMA_MODEL at this file and sets LLAMA_SPEC_TYPE=mtp.

# Option B (faster setup, no MTP): prebuilt GGUF from HuggingFace
# Comment out LLAMA_MODEL and LLAMA_SPEC_TYPE in .env first.
docker compose run --rm llama-convert download qwopus3.6-27b --quant IQ4_XS

# ── 35B standard (loads at startup) ────────────────────────────────────────

# Option A (recommended): MTP-capable GGUF from safetensors
docker compose run --rm llama-convert convert-st qwen3.6-35b-a3b --quant IQ4_XS --mtp
# Output: ./models/qwen3.6-35b-a3b-IQ4_XS-mtp.gguf

# ── 35B APEX MTP (alternative, download-only, includes MTP heads) ──────────

# APEX-MTP-I-Compact (~17.3 GB, default — fits 24 GB VRAM)
docker compose run --rm llama-convert download qwen3.6-35b-a3b-apex --quant APEX-MTP-I-Compact
# Output: ./models/qwen3.6-35b-a3b-apex-APEX-MTP-I-Compact.gguf + mmproj
# Note: To use APEX at startup, set load-on-startup = true in config/models.ini [qwen3.6-35b-a3b-apex]
#       and load-on-startup = false in [qwen3.6-35b-a3b].
```

> **Gated models:** set `HF_TOKEN=your_token` in `.env`

### Step 5: Start the services

```bash
docker compose up -d
```

> **Note:** `docker compose up -d` starts `llama-server` and `mcp-search-server` by default. `cloudflared` requires `--profile cloudflare`, and `llama-convert` requires `--profile convert`.

Check the logs:

```bash
docker compose logs -f llama-server   # up to 5 min for large model
docker compose logs -f cloudflared     # should show "connected"
```

### Step 6: Test the API

```bash
# Local test (no auth needed)
curl http://localhost:8080/health

# Public API test (requires Cloudflare Access auth)
curl -H "CF-Access-Client-Id: <id>" \
     -H "CF-Access-Client-Secret: <secret>" \
     https://api.jessfan.com/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "qwopus3.6-27b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Step 7: Connect an OpenAI-compatible client

```python
import openai

client = openai.OpenAI(
    base_url="https://api.jessfan.com/v1",
    api_key="none",  # Cloudflare Access handles auth
    default_headers={
        "CF-Access-Client-Id": "<your-client-id>",
        "CF-Access-Client-Secret": "<your-client-secret>",
    },
)

response = client.chat.completions.create(
    model="qwopus3.6-27b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Model management

```bash
# List all supported models
docker compose run --rm llama-convert list

# ── 27B (default) ──────────────────────────────────────────────────────────

# MTP-capable GGUF (recommended — from safetensors, includes nextn heads)
docker compose run --rm llama-convert convert-st qwopus3.6-27b --quant IQ4_XS --mtp
# Output: ./models/qwopus3.6-27b-IQ4_XS-mtp.gguf

# ── 35B standard ───────────────────────────────────────────────────────────

# MTP-capable GGUF from safetensors
docker compose run --rm llama-convert convert-st qwen3.6-35b-a3b --quant IQ4_XS --mtp
# Output: ./models/qwen3.6-35b-a3b-IQ4_XS-mtp.gguf

# Standard prebuilt GGUF (no MTP)
docker compose run --rm llama-convert download qwen3.6-35b-a3b --quant IQ4_XS
# Output: ./models/qwen3.6-35b-a3b-IQ4_XS.gguf

# ── 35B APEX MTP (mudler mixed-precision, download-only, MTP included) ─────

# APEX-MTP-I-Compact (~17.3 GB, recommended — fits 24 GB VRAM)
docker compose run --rm llama-convert download qwen3.6-35b-a3b-apex --quant APEX-MTP-I-Compact
# Output: ./models/qwen3.6-35b-a3b-apex-APEX-MTP-I-Compact.gguf + mmproj

# ── Re-quantize an existing GGUF already in ./models ───────────────────────
docker compose run --rm llama-convert convert /models/qwopus3.6-27b-fp16.gguf --quant Q4_K_M
```

> **APEX MTP quants** are pre-built by [mudler](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF) and use mixed-precision packing with embedded MTP heads. They cannot be produced locally — only downloaded. `APEX-MTP-I-*` variants use imatrix quantization for better quality.

> **TriAttention calibration** runs automatically after every `download` and `convert-st` — no extra flags needed.
> `calibration-data/wikitext-2-raw-test.txt` (~313k tokens, Wikipedia prose) is mounted into the container by default via `TRIATTENTION_INPUT` in `docker-compose.yml`.
> Stats are written to `./models/{model}-triattention.bin` and reused on subsequent runs (skipped if already present).
> To skip calibration entirely, set `TRIATTENTION_INPUT=` (empty) in `.env`.

---

## Docker Compose services

| Service             | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| `llama-server`      | llama.cpp inference server (port 8080)              |
| `cloudflared`       | Cloudflare Tunnel — exposes llama-server publicly   |
| `llama-convert`     | Model conversion tool (download, convert, quantize) |
| `mcp-search-server` | Web search MCP tool (port 3100)                     |

---

## Troubleshooting

- **Model not loading:** Check `docker compose logs llama-server`. Common causes: model file missing (`LLAMA_MODEL` path mismatch), insufficient VRAM.
- **MTP not working:** Confirm the GGUF was built with `--mtp`. Prebuilt GGUFs strip MTP heads. Verify `LLAMA_SPEC_TYPE=mtp` and `LLAMA_MODEL` point to the `-mtp.gguf` file.
- **Slow generation (11 vs 20 tok/s):** Context may be filling up within a long conversation. MTP requires a `-mtp.gguf` file.
- **Cloudflare Tunnel not connecting:** Verify `CF_TUNNEL_TOKEN` is correct. Check `docker compose logs cloudflared`.
- **Cloudflare Access authentication failing:** Ensure the Access Application is configured for the correct domain and authentication method.
- **GPU not detected:** Verify NVIDIA Container Toolkit is installed. Check `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`.
- **WSL2 BSOD during download/quantize:** Set `CONVERT_DOWNLOAD_RATE=300M` and `CONVERT_THREADS=4` in `.env`.
