# custom-llama

A self-hosted LLM stack built around [llama.cpp (TurboQuant fork)](https://github.com/TheTom/llama-cpp-turboquant). Includes a web chat UI, CLI assistant, IDE coding assistant support, and a Tailscale-protected API — all deployable with a single `docker compose up -d`.

## Architecture

```
                        ┌──────────────────────────────────────┐
                        │           Cloudflare Edge            │
                        │  jessfan.com  │  chat.jessfan.com    │
                        └──────────────┬───────────────────────┘
                                       │ Cloudflare Tunnel (outbound)
                        ┌──────────────▼───────────────────────┐
                        │          Host Machine                 │
                        │                                       │
        webui-net ──────┤  open-webui :8080   cloudflared      │
                        │       │                               │
                        │  open-webui-pipelines :9099           │
                        └───────┬───────────────────────────────┘
                                │ llama-net (internal Docker bridge)
               ┌────────────────┼────────────────────┐
               │                │                    │
     llama-server :8080   kv-cache-proxy :8181   openclaw-gateway :18789
               │
          tailscale sidecar
               │
         Tailscale tailnet
               │
     ┌─────────┴──────────┐
     │   Tailnet devices  │
     │  (Roo Code, CLI)   │
     └────────────────────┘
```

### Services

| Service | Image | Purpose |
|---|---|---|
| `llama-server` | custom build | llama.cpp inference, OpenAI-compatible API |
| `llama-convert` | custom build | Model download / quantize / convert (run once) |
| `tailscale` | `tailscale/tailscale` | Exposes llama-server, kv-cache-proxy, and openclaw-gateway on the tailnet |
| `kv-cache-proxy` | custom build | OpenAI-compatible proxy that pins Roo Code sessions to llama.cpp KV cache slots |
| `open-webui-pipelines` | `ghcr.io/open-webui/pipelines` | KV cache slot management pipeline for the web UI |
| `open-webui` | `ghcr.io/open-webui/open-webui` | Web chat UI, gated by Google OAuth |
| `openclaw-gateway` | `ghcr.io/openclaw/openclaw` | CLI assistant (`openclaw agent`) on the tailnet |
| `cloudflared` | `cloudflare/cloudflared` | Cloudflare Tunnel — exposes Open WebUI at chat.jessfan.com, zero inbound ports |

### Access matrix

| Interface | URL / Command | Auth |
|---|---|---|
| **Local (host machine)** | `http://localhost:8080/v1` | None |
| **Web UI** | `https://chat.jessfan.com` | Google OAuth (Firebase credentials) |
| **CLI** | `openclaw --gateway http://<ts-host>:18789 agent --message "..."` | Gateway token (tailnet only) |
| **Roo Code** | `http://<ts-host>:8181/v1` + `X-Session-ID` header | Tailnet only, no auth |
| **Direct API** | `http://<ts-host>:8080/v1` | Tailnet only, no auth |

## Local-only quick start (no Tailscale / Cloudflare needed)

If you just want to run the inference server on the machine you're sitting at
and call it from `localhost`, you don't need Tailscale, Cloudflare, or any of
the web-UI secrets. Skip the external setup entirely.

**1. Configure**

```bash
cp .env.default .env
# Only required: MODEL_NAME, QUANT (and HF_TOKEN for gated models)
# Leave TS_AUTHKEY, CF_TUNNEL_TOKEN, WEBUI_SECRET_KEY, etc. blank for now
```

**2. Expose the port locally**

Add a `ports:` mapping to `llama-server` in `docker-compose.yml`, or create a
`docker-compose.override.yml` (not committed) so the change doesn't touch the
main file:

```yaml
# docker-compose.override.yml
services:
  llama-server:
    ports:
      - "8080:8080"
```

**3. Prepare a model and start**

```bash
docker compose build
docker compose build llama-convert
docker compose run --rm llama-convert download qwen3.5-9b --quant Q4_K_M

# Start only the inference server (skips Tailscale, Cloudflare, Open WebUI, etc.)
docker compose up -d llama-server
docker compose logs -f llama-server   # wait until "llama server listening"
```

**4. Call it**

```bash
curl http://localhost:8080/health

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Any OpenAI-compatible client (LM Studio, Cursor, Roo Code, Continue, etc.) can
point at `http://localhost:8080/v1` with any non-empty API key string.

## Prerequisites

- Docker + Docker Compose v2
- NVIDIA GPU (RTX 30xx or newer recommended for TurboQuant KV-cache)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- A [Tailscale](https://tailscale.com) account
- A Cloudflare account (the `jessfan.com` domain must already be on Cloudflare)

```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. One-time external setup (three tokens)

Collect these before running anything. Each takes ~2 minutes.

**Tailscale auth key**
1. Tailscale admin → Settings → Keys → Generate auth key
2. Make it ephemeral and pre-authorized; optionally tag it `tag:llama`
3. Copy the key → `TS_AUTHKEY` in `.env`

**Cloudflare Tunnel token**
1. Cloudflare Zero Trust → Networks → Tunnels → Create a tunnel → Docker
2. Copy the tunnel token → `CF_TUNNEL_TOKEN` in `.env`
3. Add a Public Hostname: `chat.jessfan.com` → `http://open-webui:8080`
   (Cloudflare creates the DNS record automatically)

**Google OAuth credentials**
1. Firebase Console → Authentication → Sign-in method → Google → Web SDK configuration
   (or Google Cloud Console → APIs & Credentials → OAuth 2.0 Client IDs)
2. Add `https://chat.jessfan.com/oauth/oidc/callback` to Authorized redirect URIs
3. Copy Client ID → `GOOGLE_CLIENT_ID` and Client Secret → `GOOGLE_CLIENT_SECRET` in `.env`

### 2. Clone and configure

```bash
git clone https://github.com/JEF1056/custom-llama.git
cd custom-llama
cp .env.default .env
```

Fill in `.env` (minimum required fields):

```bash
# Tailscale
TS_AUTHKEY=tskey-auth-...

# Cloudflare Tunnel
CF_TUNNEL_TOKEN=eyJ...

# Google OAuth
GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-...

# Secrets — generate with: openssl rand -hex 32
WEBUI_SECRET_KEY=...
OPENCLAW_GATEWAY_TOKEN=...
PIPELINES_API_KEY=...
```

### 3. Prepare a model

```bash
docker compose build
docker compose build llama-convert  # builds the convert image (profile: convert)

# Download a pre-built GGUF (most models)
docker compose run --rm llama-convert download qwen3.5-27b --quant Q4_K_M

# Safetensors-only repo: convert → fp16 GGUF → quantize
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0
```

> **Gated / private models:** set `HF_TOKEN=your_token` in `.env` before running the convert image.

### 4. Start the stack

```bash
docker compose up -d
```

All eight services start. The server takes up to 5 minutes to load a large model;
`docker compose logs -f llama-server` shows progress.

### 5. Verify

```bash
# Web UI
open https://chat.jessfan.com

# Tailnet API (from a tailnet machine)
curl http://llama-api.<tailnet>.ts.net:8080/health
```

## Configuring Access Methods

Each interface requires a one-time setup on the **client machine** (not the server).
Replace `llama-api.<tailnet>.ts.net` with your actual Tailscale machine name
(find it at [login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines)).

---

### Open WebUI (web browser)

**Who it's for:** anyone with a browser who you want to give chat access to.
**Auth:** Google OAuth (Firebase credentials).
**URL:** `https://chat.jessfan.com`

#### Server-side setup (done once after `docker compose up -d`)

**1. Connect the KV-cache pipeline**

Open WebUI automatically connects to the pipelines server via `OPENAI_API_BASE_URL`
(set in `docker-compose.yml`). The pipeline in `pipelines/kv_cache.py` is
auto-loaded when the pipelines container starts. Verify it appeared:

1. Sign in to `https://chat.jessfan.com` with your Google account
   (the first account becomes admin automatically)
2. Admin panel → Settings → Connections — confirm `http://open-webui-pipelines:9099`
   appears under OpenAI connections
3. Admin panel → Pipelines — confirm **Llama KV Cache** is listed and enabled

**2. Lock registration** (after adding all allowed accounts)

Set `ENABLE_OAUTH_SIGNUP=false` in `.env` and restart Open WebUI:

```bash
docker compose restart open-webui
```

Only accounts that have already signed in can access the UI after this.

**3. Verify the model appears**

Admin panel → Models — you should see the model name returned by llama.cpp
(e.g. `qwopus3.6-35b-TQ2_0`). If it shows "Llama (loading…)" the server is
still loading the model; wait and refresh.

#### Using the web UI

- Start a new chat → select the model → chat normally
- Each conversation is automatically pinned to a persistent KV cache slot
- Slot state is saved to `./models/slots/` so context survives server restarts
- The reasoning `<think>` blocks are preserved in context across turns
  (`LLAMA_PRESERVE_THINKING=on`)

---

### OpenClaw CLI

**Who it's for:** yourself — ad-hoc AI queries from any terminal, including the
VSCode integrated terminal.
**Auth:** gateway token (tailnet only).
**Prerequisite:** machine must be on the Tailscale tailnet.

#### Installation (client machine, run once)

```bash
# Requires Node 24 (recommended) or Node 22.16+
npm install -g openclaw@latest

# Verify
openclaw --version
```

#### Client configuration

Create `~/.openclaw/openclaw.json` on each client machine:

```json
{
  "gateway": {
    "url": "http://llama-api.<tailnet>.ts.net:18789",
    "token": "<OPENCLAW_GATEWAY_TOKEN from .env>"
  }
}
```

Alternatively, pass flags inline (useful for scripts):

```bash
openclaw \
  --gateway "http://llama-api.<tailnet>.ts.net:18789" \
  --token   "<OPENCLAW_GATEWAY_TOKEN>" \
  agent --message "hello"
```

Or set environment variables in your shell profile:

```bash
export OPENCLAW_GATEWAY_URL="http://llama-api.<tailnet>.ts.net:18789"
export OPENCLAW_GATEWAY_TOKEN="<OPENCLAW_GATEWAY_TOKEN>"
```

#### Common commands

```bash
# One-shot query
openclaw agent --message "explain this function"

# Interactive chat session (press Ctrl-C to exit)
openclaw agent

# Pipe context in from stdin
cat myfile.py | openclaw agent --message "review this code"

# With extended reasoning
openclaw agent --message "design a caching strategy" --thinking high

# Check gateway health
openclaw gateway health

# View active sessions
openclaw sessions list
```

#### VSCode integration

No extension needed. Open the VSCode integrated terminal (`Ctrl+`` ` ``) and run
`openclaw agent` commands directly. Because the client config lives in
`~/.openclaw/openclaw.json`, it works in any project without extra setup.

---

### Roo Code (VSCode extension)

**Who it's for:** yourself — AI-assisted coding inside VSCode.
**Auth:** none required (tailnet only).
**Prerequisite:** machine must be on the Tailscale tailnet.

Roo Code routes through the `kv-cache-proxy` (port 8181) instead of directly
to llama.cpp. The proxy pins each workspace to a fixed KV cache slot via the
`X-Session-ID` header, so the model retains full conversation context including
Roo Code's Intelligent Context Condensing summaries.

#### Provider configuration

Open VSCode → Roo Code settings (`⌘,` → search "Roo") → **API Provider** → **OpenAI Compatible**.

| Setting | Value |
|---|---|
| **Base URL** | `http://llama-api.<tailnet>.ts.net:8181/v1` |
| **API Key** | `none` (any non-empty string) |
| **Model ID** | `qwopus3.6-35b-TQ2_0` *(or whichever model is loaded)* |

Or edit `.roo/config.json` (or `~/.roo/config.json` for global):

```json
{
  "apiProvider": "openai-compatible",
  "openAiBaseUrl": "http://llama-api.<tailnet>.ts.net:8181/v1",
  "openAiApiKey": "none",
  "openAiModelId": "qwopus3.6-35b-TQ2_0"
}
```

#### Session ID (KV cache pinning)

Add a custom header so the proxy assigns this workspace its own persistent slot.
In Roo Code settings → **Custom Headers**:

| Header name | Value |
|---|---|
| `X-Session-ID` | `roo-<your-workspace-name>` |

Use a consistent, unique name per workspace (e.g. `roo-custom-llama`,
`roo-portfolio`). The proxy maps this string to a slot number and restores
that slot's KV cache before every request, so the model never cold-starts
mid-conversation even after Roo Code's context condensing runs.

#### Context condensing

Leave Roo Code's **Intelligent Context Condensing** enabled (it's on by default).
When the context window fills up, Roo Code summarizes earlier turns using the
same endpoint — the proxy handles the slot pinning transparently, so the
condensed summary lands in the same KV cache slot and the model retains full
context continuity.

Recommended threshold (Roo Code settings → Context):

| Setting | Recommended value |
|---|---|
| Automatically trigger condensing | ✅ enabled |
| Threshold | 80% |

#### Reasoning model settings

The model is a reasoning model (chain-of-thought). In Roo Code settings:

| Setting | Value |
|---|---|
| **Enable thinking / extended reasoning** | on |
| **Thinking budget** | 10 000–20 000 tokens (adjust to taste) |

---

### Direct API (tailnet)

For scripts, other tools, or testing. No auth required on the tailnet.

```bash
# Health check
curl http://llama-api.<tailnet>.ts.net:8080/health

# Chat completion (non-streaming)
curl http://llama-api.<tailnet>.ts.net:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Streaming
curl http://llama-api.<tailnet>.ts.net:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Pin to a specific slot and cache the prompt
curl http://llama-api.<tailnet>.ts.net:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "id_slot": 0,
    "cache_prompt": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## KV Cache / Session Management

llama.cpp serves `LLAMA_PARALLEL` concurrent inference slots (default: 2). Each slot
has its own KV cache, giving each session an independent 100K-token context.

The stack adds two layers of slot management:

- **Open WebUI pipeline** (`pipelines/kv_cache.py`): maps `chat_id` → slot, calls
  `/slots/{id}?action=restore` before each request and `?action=save` after
- **kv-cache-proxy** (`kv-cache-proxy/main.py`): same lifecycle, keyed off the
  `X-Session-ID` request header (for Roo Code and other CLI tools)

With `LLAMA_SLOT_SAVE_PATH=/models/slots` (default), slot state persists to disk
so KV caches survive server restarts.

## OpenClaw Config

`openclaw/openclaw.json` configures the Gateway and its llama.cpp provider. Update
the model `id` and `name` when you change `MODEL_NAME` or `QUANT` in `.env`:

```json
{
  "models": {
    "providers": {
      "llamacpp": {
        "models": [{ "id": "qwopus3.6-35b-TQ2_0", "name": "Qwopus 3.6 35B (TQ2_0)", ... }]
      }
    }
  }
}
```

## Configuration Reference

Copy `.env.default` to `.env` and adjust. All `LLAMA_*` defaults are tuned for an
RTX 3090 (24 GB VRAM) desktop.

### Model selection

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `qwopus3.6-35b` | Key from the models list |
| `QUANT` | `TQ2_0` | Quantization type |
| `LLAMA_MODEL` | _(empty)_ | Override: explicit path to a `.gguf` in `/models` |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_HOST` | `0.0.0.0` | Bind address (inside the container / llama-net) |
| `LLAMA_PORT` | `8080` | Bind port |
| `LLAMA_GPU_LAYERS` | `99` | Layers to offload to GPU (clamped to model max) |
| `LLAMA_CTX_SIZE` | `200000` | Total context pool (shared across parallel slots) |
| `LLAMA_PARALLEL` | `2` | Concurrent inference slots |
| `LLAMA_THREADS` | `6` | CPU threads for decode |
| `LLAMA_THREADS_BATCH` | `12` | CPU threads for prompt prefill |
| `LLAMA_BATCH_SIZE` | `4096` | Logical batch size |
| `LLAMA_UBATCH_SIZE` | `1024` | Physical micro-batch per CUDA kernel |
| `LLAMA_MAX_TOKENS` | `-1` | Max tokens per response (`-1` = unlimited) |
| `LLAMA_TEMP` | `0.7` | Sampling temperature |
| `LLAMA_TOP_P` | `0.95` | Top-p sampling |
| `LLAMA_NO_MMAP` | `off` | `on` = load model into RAM; `off` = mmap (default) |

### KV cache

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CACHE_TYPE_K` | `turbo3` | K-cache type: `f16`, `turbo3`, `turbo4` |
| `LLAMA_CACHE_TYPE_V` | `turbo3` | V-cache type: `f16`, `turbo3`, `turbo4` |
| `LLAMA_FLASH_ATTN` | `on` | Required for TurboQuant KV cache |
| `LLAMA_NO_KV_OFFLOAD` | `off` | `on` = keep KV cache on GPU at all times |
| `LLAMA_SLOT_SAVE_PATH` | `/models/slots` | Directory for persistent slot state (save/restore) |

### Reasoning / chat

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_REASONING` | `on` | Chain-of-thought output |
| `LLAMA_PRESERVE_THINKING` | `on` | Include prior `<think>` blocks in context |

### Networking / auth

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_API_KEY` | _(empty)_ | Bearer token for the raw llama.cpp API (leave empty on tailnet) |
| `TS_AUTHKEY` | _(required)_ | Tailscale auth key |
| `TS_HOSTNAME` | `llama-api` | Tailscale hostname for this machine |
| `CF_TUNNEL_TOKEN` | _(required)_ | Cloudflare Tunnel token |

### Web UI

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBUI_SECRET_KEY` | _(required)_ | Session signing key — `openssl rand -hex 32` |
| `WEBUI_URL` | `https://chat.jessfan.com` | Public URL (used for OAuth redirect) |
| `GOOGLE_CLIENT_ID` | _(required)_ | Google OAuth Client ID |
| `GOOGLE_CLIENT_SECRET` | _(required)_ | Google OAuth Client Secret |
| `ENABLE_OAUTH_SIGNUP` | `true` | `false` = only pre-existing accounts can sign in |
| `PIPELINES_API_KEY` | _(required)_ | Shared secret between Open WebUI and the pipelines server |

### OpenClaw

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCLAW_GATEWAY_TOKEN` | _(required)_ | Bearer token for CLI clients |
| `KV_PROXY_PORT` | `8181` | Port for the kv-cache-proxy |

### Multi-GPU / MoE

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_TS` | _(empty)_ | Tensor split (e.g. `13,14` for two GPUs) |
| `LLAMA_NCMOE` | _(empty)_ | MoE experts to offload to CPU |

### Multimodal

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MMPROJ` | `/models/mmproj.gguf` | Path to the multimodal projector |
| `LLAMA_IMAGE` | _(empty)_ | Path to a static image pre-loaded at startup |

### HuggingFace

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | _(empty)_ | Token for gated/private model repos |

## Available Models

Run `docker compose run --rm llama-convert list` to see the full list with sizes.

### Small (<4 GB at Q4_K_M)

| Key | Description |
|-----|-------------|
| `qwen3.5-0.8b` | Qwen 3.5 0.8B (~0.5 GB) |
| `llama3.2-1b` | Llama 3.2 1B Instruct (~0.7 GB) |
| `llama3.2-3b` | Llama 3.2 3B Instruct (~2 GB) |
| `gemma-4-e2b` | Gemma 4 E2B (~1.5 GB) **[Multimodal]** |
| `qwen3.5-4b` | Qwen 3.5 4B (~2.5 GB) |

### Medium (4–12 GB at Q4_K_M)

| Key | Description |
|-----|-------------|
| `qwen2.5-coder-7b` | Qwen 2.5 Coder 7B Instruct (~4.5 GB) |
| `gemma-4-e4b` | Gemma 4 E4B (~3 GB) **[Multimodal]** |
| `qwen3.5-9b` | Qwen 3.5 9B (~5.5 GB) |
| `gpt-oss-20b` | GPT-OSS 20B (~11 GB) |

### Large (12–18 GB at Q4_K_M)

| Key | Description |
|-----|-------------|
| `gemma-4-26b-a4b` | Gemma 4 26B-A4B (~13 GB) **[Multimodal]** |
| `gemma-4-31b` | Gemma 4 31B (~16 GB) **[Multimodal]** |
| `qwen3.6-27b` | Qwen 3.6 27B (~14 GB) |
| `qwen3.5-27b` | Qwen 3.5 27B (~14 GB) |
| `qwen3.6-35b-a3b` | Qwen 3.6 35B-A3B (~17 GB) |
| `qwopus3.6-35b` | Qwopus 3.6 35B-A3B-v1 (~17 GB) **[Multimodal]** |
| `minimax-m2.7` | MiniMax M2.7 (~18 GB) |

## Quantization Guide

### Standard quants

Downloaded directly from HuggingFace when available; otherwise quantized locally.

| Quant | Size vs Q4_K_M | Notes |
|-------|----------------|-------|
| `Q4_K_M` | 1× | Recommended default — best quality/size balance |
| `Q5_K_M` | 1.2× | Slightly higher quality |
| `Q6_K` | 1.5× | High quality |
| `Q8_0` | 2× | Near-lossless; good intermediate for re-quantization |
| `Q3_K_M` | 0.75× | Smaller, some quality loss |
| `IQ4_XS` | 0.9× | Imatrix-optimized 4-bit |

### TurboQuant (TQ2_0 / TQ1_0)

TurboQuant requires **fp16 or bf16 GGUF** as the source.

| Quant | Bits/weight | Notes |
|-------|-------------|-------|
| `TQ2_0` | ~2 | Better quality, ~4–5× smaller than fp16 |
| `TQ1_0` | ~1 | Maximum compression, some quality loss |

```bash
# fp16 GGUF exists on HuggingFace (auto-download + quantize)
docker compose run --rm llama-convert download qwen3.6-27b --quant TQ2_0

# safetensors only (downloads, converts, quantizes)
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0
```

## Model Management Commands

```bash
# List all supported models with sizes
docker compose run --rm llama-convert list

# Download a pre-built GGUF (or quantize locally if quant not on HF)
docker compose run --rm llama-convert download <model> --quant <quant>

# Convert safetensors → fp16 GGUF → target quant
docker compose run --rm llama-convert convert-st <model> --quant <quant>

# Re-quantize an existing GGUF already in ./models
docker compose run --rm llama-convert convert /models/model-Q8_0.gguf --quant Q4_K_M
```

## Common Operations

```bash
# Start full stack
docker compose up -d

# Start only the inference server (no web UI / tunnels)
docker compose up -d llama-server tailscale

# View logs
docker compose logs -f llama-server
docker compose logs -f open-webui
docker compose logs -f cloudflared

# Stop everything
docker compose down

# Rebuild the llama-server image
docker compose up -d --build llama-server

# Switch models: edit MODEL_NAME/QUANT in .env, update openclaw/openclaw.json, then restart
docker compose restart llama-server openclaw-gateway
```

## API Reference

The server exposes an OpenAI-compatible HTTP API. Accessible via:
- `llama-net` internally (pipelines, kv-cache-proxy)
- `http://<ts-hostname>:8080` on the tailnet (via tailscale sidecar)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List loaded model |
| `POST` | `/v1/chat/completions` | Chat completions |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/embeddings` | Embeddings |
| `GET` | `/slots` | List inference slots |
| `POST` | `/slots/{id}?action=save` | Save slot KV cache to disk |
| `POST` | `/slots/{id}?action=restore` | Restore slot KV cache from disk |

## Multimodal (Vision)

Models tagged **[Multimodal]** download their `mmproj.gguf` automatically.

```bash
docker compose run --rm llama-convert download qwopus3.6-35b --quant TQ2_0
# ./models/mmproj.gguf is placed automatically

# Verify LLAMA_MMPROJ=/models/mmproj.gguf in .env (it is the default)
docker compose up -d
```

Send an image via the API:

```bash
curl http://llama-api.<tailnet>.ts.net:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
      ]
    }]
  }'
```

## Security

### Threat model

| Attacker | Capability | Mitigation |
|---|---|---|
| Public internet user | Can reach `chat.jessfan.com` | Must complete Google OAuth; rejected if no account exists |
| Google account holder (not allowed) | Can attempt OAuth | `ENABLE_OAUTH_SIGNUP=false` (default) prevents account creation; existing-account check rejects unknowns |
| Tailnet outsider | Cannot reach ports 8080 / 8181 / 18789 | All three are TCP-forwarded by the Tailscale sidecar — no host port mappings, no public exposure |
| Tailnet member | Can reach llama.cpp API directly | Intentional; Tailscale ACLs restrict *which* tailnet devices can reach `tag:llama` |
| Compromised `open-webui` container | On `webui-net` only | Cannot reach `llama-net` services directly — `llama-net: internal: true` and no cross-network route |
| Compromised `open-webui-pipelines` | Bridges both networks | Only callable with a strong `PIPELINES_API_KEY`; can reach llama-server but not the internet (llama-net is internal) |

### Network isolation

```
Internet ──► cloudflared ──► open-webui ──► open-webui-pipelines ──► llama-server
                                                                  └──► kv-cache-proxy

ts-net (internet ✓): tailscale only
webui-net (internet ✓): cloudflared, open-webui, open-webui-pipelines (webui side)
llama-net (internal ✗): tailscale, llama-server, kv-cache-proxy, openclaw-gateway,
                         open-webui-pipelines (llama side)
```

Services on `llama-net` have no route to the public internet. The Tailscale sidecar
is the only container that bridges `llama-net` and the internet (via `ts-net`), which
it needs to reach the Tailscale coordination server.

### Tailscale ACLs

Configure your Tailscale ACL policy (admin console → Access controls) to restrict
which devices can reach the inference ports. Example policy:

```json
{
  "tagOwners": {
    "tag:llama": ["autogroup:admin"]
  },
  "acls": [
    {
      "action": "accept",
      "src":    ["autogroup:admin"],
      "dst":    ["tag:llama:8080", "tag:llama:8181", "tag:llama:18789"]
    }
  ]
}
```

Then tag the Tailscale auth key with `tag:llama` when generating it in
`TS_EXTRA_ARGS=--advertise-tags=tag:llama`. Devices not in `autogroup:admin`
will be denied even if they are on the tailnet.

### First-run account creation

`ENABLE_OAUTH_SIGNUP` defaults to `false`. Open WebUI special-cases the
very first sign-in: when no accounts exist in the database, the first Google
OAuth login creates an admin account **regardless of this setting**. After that,
new signups are blocked.

To add another allowed account:
1. Set `ENABLE_OAUTH_SIGNUP=true` in `.env` and `docker compose restart open-webui`
2. Have the user sign in once (account is created)
3. Set it back to `false` and restart again

### Secret checklist

All five secrets must be set in `.env` before running `docker compose up`. The
compose file will refuse to start with a clear error if any are missing.

```bash
# Generate all secrets at once
echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)"
echo "OPENCLAW_GATEWAY_TOKEN=$(openssl rand -hex 32)"
echo "PIPELINES_API_KEY=$(openssl rand -hex 16)"
```

| Variable | Purpose | Generation |
|---|---|---|
| `WEBUI_SECRET_KEY` | Signs Open WebUI session cookies | `openssl rand -hex 32` |
| `PIPELINES_API_KEY` | Authenticates Open WebUI → pipelines | `openssl rand -hex 16` |
| `OPENCLAW_GATEWAY_TOKEN` | Authenticates CLI → OpenClaw Gateway | `openssl rand -hex 32` |
| `TS_AUTHKEY` | Joins Tailscale tailnet | Tailscale admin console |
| `CF_TUNNEL_TOKEN` | Authenticates Cloudflare Tunnel | Cloudflare Zero Trust |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth | Firebase / Google Cloud Console |

---

## Troubleshooting

### Model file not found at startup

```
ERROR: Model file not found: /models/qwopus3.6-35b-TQ2_0.gguf
```

Run the prepare step first:

```bash
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0
```

### TurboQuant: no fp16/bf16 GGUF source

```
Error: No fp16 or bf16 GGUF found in <repo>.
```

Use `convert-st` to download safetensors and convert:

```bash
docker compose run --rm llama-convert convert-st <model> --quant TQ2_0
```

### Out of VRAM

```bash
# In .env
LLAMA_CTX_SIZE=65536
LLAMA_PARALLEL=1
# or
LLAMA_GPU_LAYERS=40
```

### Cloudflare Tunnel not connecting

```bash
docker compose logs -f cloudflared
```

Ensure `CF_TUNNEL_TOKEN` is set and the Public Hostname in Zero Trust points to `http://open-webui:8080`.

### Tailscale not joining tailnet

```bash
docker compose logs -f tailscale
```

Ensure `TS_AUTHKEY` is set and not expired. Auth keys are single-use by default — generate a new one if the container was ever restarted with the same key after first use.

### Open WebUI shows "connection error" to pipelines

The pipelines server starts after llama-server, which takes up to 5 minutes to load the model. Wait for `docker compose logs llama-server` to show the model is loaded, then reload Open WebUI.

### GPU not visible

```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Install toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## License

This project is provided as-is. llama.cpp is licensed under MIT — see [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp). OpenClaw is licensed under MIT — see [openclaw/openclaw](https://github.com/openclaw/openclaw).
