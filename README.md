# custom-llama

A self-hosted LLM stack built around [llama.cpp (TurboQuant fork)](https://github.com/TheTom/llama-cpp-turboquant). Web chat UI, CLI assistant, and IDE coding support — all deployable with `docker compose up -d`.

## Local quick start

No Tailscale, Cloudflare, or secrets needed — just the inference server on this machine.

```bash
cp .env.default .env   # set MODEL_NAME + QUANT; leave everything else blank
docker compose build && docker compose build llama-convert
docker compose run --rm llama-convert download qwen3.5-9b --quant Q4_K_M
docker compose up -d llama-server
```

Port 8080 is not exposed by default. Create `docker-compose.override.yml` (gitignored) to open it on localhost:

```yaml
services:
  llama-server:
    ports:
      - "8080:8080"
```

Then `docker compose up -d llama-server` (not `restart` — that won't re-read the config). Test with `curl http://localhost:8080/health`.

Any OpenAI-compatible client (Cursor, Roo Code, LM Studio, etc.) points at `http://localhost:8080/v1`.

---

## Architecture

```
                  ┌──────────────────────────────────┐
                  │         Cloudflare Edge           │
                  │  jessfan.com  │  chat.jessfan.com │
                  └──────────────┬────────────────────┘
                                 │ Cloudflare Tunnel (outbound)
                  ┌──────────────▼────────────────────┐
                  │         Host Machine               │
                  │  cloudflared → open-webui :8080   │  webui-net
                  │            open-webui-pipelines    │
                  └───────────────┬────────────────────┘
                                  │ llama-net (internal, no internet)
             ┌────────────────────┼──────────────────┐
             │                    │                  │
   llama-server :8080   kv-cache-proxy :8181   openclaw-gateway :18789
             │
        tailscale sidecar → Tailscale tailnet → your devices (Roo Code, CLI)
```

| Interface | URL / Command | Auth |
|---|---|---|
| **Local** | `http://localhost:8080/v1` | None (requires `docker-compose.override.yml`) |
| **Web UI** | `https://chat.jessfan.com` | Google OAuth |
| **CLI** | `openclaw agent --message "..."` | Gateway token (tailnet only) |
| **Roo Code** | `http://<ts-host>:8181/v1` | Tailnet only |
| **Direct API** | `http://<ts-host>:8080/v1` | Tailnet only |

## Full stack deployment

### Prerequisites

- Docker + Docker Compose v2, NVIDIA GPU, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Tailscale account, Cloudflare account (domain already on Cloudflare)

### 1. Collect three tokens

**Tailscale:** admin → Settings → Keys → Generate auth key (ephemeral, pre-authorized, tag `tag:llama`) → `TS_AUTHKEY`

**Cloudflare Tunnel:** Zero Trust → Networks → Tunnels → Create → Docker → copy token → `CF_TUNNEL_TOKEN`. Add Public Hostname: `chat.jessfan.com` → `http://open-webui:8080`

**Google OAuth:** Firebase Console → Authentication → Sign-in method → Google → Web SDK config (or Google Cloud Console → OAuth 2.0 Client IDs). Add redirect URI `https://chat.jessfan.com/oauth/oidc/callback` → `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET`

### 2. Configure

```bash
cp .env.default .env
```

Minimum required fields:

```bash
TS_AUTHKEY=tskey-auth-...
CF_TUNNEL_TOKEN=eyJ...
GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-...

WEBUI_SECRET_KEY=$(openssl rand -hex 32)
OPENCLAW_GATEWAY_TOKEN=$(openssl rand -hex 32)
PIPELINES_API_KEY=$(openssl rand -hex 16)
```

### 3. Prepare a model

```bash
docker compose build
docker compose build llama-convert

# Most models (pre-built GGUF on HuggingFace)
docker compose run --rm llama-convert download qwen3.5-27b --quant Q4_K_M

# Safetensors-only repos (convert → fp16 → quantize)
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0
```

> **Gated models:** set `HF_TOKEN=your_token` in `.env`

### 4. Start

```bash
docker compose up -d
docker compose logs -f llama-server   # up to 5 min for large model
```

## Configuring access

Replace `<ts-host>` with your machine's Tailscale hostname from [login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines).

### Open WebUI

Sign in at `https://chat.jessfan.com` with Google — the first login creates an admin account automatically. After creating all allowed accounts, set `ENABLE_OAUTH_SIGNUP=false` in `.env` and `docker compose restart open-webui` to lock registration.

Verify setup: Admin panel → Pipelines → confirm **Llama KV Cache** is listed.

### OpenClaw CLI

```bash
npm install -g openclaw@latest   # Node 24+
```

Configure in `~/.openclaw/openclaw.json`:

```json
{
  "gateway": {
    "url": "http://<ts-host>:18789",
    "token": "<OPENCLAW_GATEWAY_TOKEN>"
  }
}
```

```bash
openclaw agent --message "explain this"   # one-shot
openclaw agent                            # interactive
cat file.py | openclaw agent --message "review this"
openclaw agent --message "design this" --thinking high
```

Works in the VSCode integrated terminal without any extension.

### Roo Code

In Roo Code settings → **API Provider** → **OpenAI Compatible**:

| Setting | Value |
|---|---|
| Base URL | `http://<ts-host>:8181/v1` |
| API Key | `none` |
| Model ID | `qwopus3.6-35b-TQ2_0` |

Add a custom header to pin the workspace to a persistent KV cache slot:

| Header | Value |
|---|---|
| `X-Session-ID` | `roo-<workspace-name>` |

Use a unique name per workspace (e.g. `roo-custom-llama`). The proxy restores that slot's KV cache before every request, including Intelligent Context Condensing calls.

## Security

| Attacker | Mitigation |
|---|---|
| Public user without Google account | `ENABLE_OAUTH_SIGNUP=false` — no self-registration |
| Google account holder not on the allowlist | Open WebUI rejects unknown accounts |
| Non-tailnet user trying to hit the API | No public port mappings; Tailscale-only |
| Tailnet member not in ACL | Tailscale ACL restricts `tag:llama` ports to `autogroup:admin` |
| Compromised `open-webui` container | On `webui-net` only — no route to `llama-net` (internal) |

### Tailscale ACL

```json
{
  "tagOwners": { "tag:llama": ["autogroup:admin"] },
  "acls": [{
    "action": "accept",
    "src": ["autogroup:admin"],
    "dst": ["tag:llama:8080", "tag:llama:8181", "tag:llama:18789"]
  }]
}
```

Set `TS_EXTRA_ARGS=--advertise-tags=tag:llama` in `.env`.

## Configuration reference

All `LLAMA_*` defaults are tuned for RTX 3090 (24 GB VRAM).

### Model

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `qwopus3.6-35b` | Model key |
| `QUANT` | `TQ2_0` | Quantization |
| `LLAMA_MODEL` | — | Override: explicit `.gguf` path in `/models` |

### Server

| Variable | Default | Description |
|---|---|---|
| `LLAMA_PORT` | `8080` | Bind port |
| `LLAMA_GPU_LAYERS` | `99` | GPU layers (clamped to model max) |
| `LLAMA_CTX_SIZE` | `200000` | Total context pool across all slots |
| `LLAMA_PARALLEL` | `2` | Concurrent inference slots |
| `LLAMA_THREADS` | `6` | CPU decode threads |
| `LLAMA_THREADS_BATCH` | `12` | CPU prefill threads |
| `LLAMA_BATCH_SIZE` | `4096` | Logical batch size |
| `LLAMA_UBATCH_SIZE` | `1024` | Physical micro-batch per CUDA kernel |
| `LLAMA_MAX_TOKENS` | `-1` | Max tokens per response |
| `LLAMA_TEMP` | `0.7` | Temperature |
| `LLAMA_TOP_P` | `0.95` | Top-p |
| `LLAMA_NO_MMAP` | `off` | `on` = load model into RAM |

### KV cache

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CACHE_TYPE_K` | `turbo3` | `f16`, `turbo3`, `turbo4` |
| `LLAMA_CACHE_TYPE_V` | `turbo3` | `f16`, `turbo3`, `turbo4` |
| `LLAMA_FLASH_ATTN` | `on` | Required for TurboQuant |
| `LLAMA_SLOT_SAVE_PATH` | `/models/slots` | Persist slot state across restarts |

### Reasoning

| Variable | Default | Description |
|---|---|---|
| `LLAMA_REASONING` | `on` | Chain-of-thought output |
| `LLAMA_PRESERVE_THINKING` | `on` | Keep `<think>` blocks in context across turns |

### Stack secrets

| Variable | Description |
|---|---|
| `TS_AUTHKEY` | Tailscale auth key |
| `CF_TUNNEL_TOKEN` | Cloudflare Tunnel token |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth |
| `WEBUI_SECRET_KEY` | Open WebUI session signing key |
| `PIPELINES_API_KEY` | Open WebUI → pipelines auth |
| `OPENCLAW_GATEWAY_TOKEN` | CLI → OpenClaw Gateway auth |

### Other

| Variable | Default | Description |
|---|---|---|
| `LLAMA_API_KEY` | — | Bearer token for raw API (empty = open on tailnet) |
| `TS_HOSTNAME` | `llama-api` | Tailscale machine name |
| `LLAMA_TS` | — | Tensor split for multi-GPU (e.g. `13,14`) |
| `LLAMA_NCMOE` | — | MoE experts to offload to CPU |
| `LLAMA_MMPROJ` | `/models/mmproj.gguf` | Multimodal projector path |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## Available models

Run `docker compose run --rm llama-convert list` for the full list with sizes.

| Key | Description |
|---|---|
| `qwen3.5-0.8b` | Qwen 3.5 0.8B (~0.5 GB) |
| `llama3.2-1b` | Llama 3.2 1B Instruct (~0.7 GB) |
| `llama3.2-3b` | Llama 3.2 3B Instruct (~2 GB) |
| `gemma-4-e2b` | Gemma 4 E2B (~1.5 GB) **[Multimodal]** |
| `qwen3.5-4b` | Qwen 3.5 4B (~2.5 GB) |
| `qwen2.5-coder-7b` | Qwen 2.5 Coder 7B Instruct (~4.5 GB) |
| `gemma-4-e4b` | Gemma 4 E4B (~3 GB) **[Multimodal]** |
| `qwen3.5-9b` | Qwen 3.5 9B (~5.5 GB) |
| `gpt-oss-20b` | GPT-OSS 20B (~11 GB) |
| `gemma-4-26b-a4b` | Gemma 4 26B-A4B (~13 GB) **[Multimodal]** |
| `qwen3.6-27b` / `qwen3.5-27b` | Qwen 3.6/3.5 27B (~14 GB) |
| `gemma-4-31b` | Gemma 4 31B (~16 GB) **[Multimodal]** |
| `qwen3.6-35b-a3b` | Qwen 3.6 35B-A3B (~17 GB) |
| `qwopus3.6-35b` | Qwopus 3.6 35B-A3B-v1 (~17 GB) **[Multimodal]** |
| `minimax-m2.7` | MiniMax M2.7 (~18 GB) |

## Quantization

| Quant | Bits | Notes |
|---|---|---|
| `Q4_K_M` | ~4 | Recommended default |
| `Q5_K_M` | ~5 | Higher quality |
| `Q8_0` | ~8 | Near-lossless; good source for re-quantization |
| `TQ2_0` | ~2 | TurboQuant — requires fp16/bf16 source |
| `TQ1_0` | ~1 | TurboQuant maximum compression |

TurboQuant needs an fp16/bf16 GGUF as source. If the HuggingFace repo is safetensors-only, use `convert-st` instead of `download`.

## Model management

```bash
docker compose run --rm llama-convert list
docker compose run --rm llama-convert download <model> --quant <quant>
docker compose run --rm llama-convert convert-st <model> --quant <quant>  # safetensors source
docker compose run --rm llama-convert convert /models/model-Q8_0.gguf --quant Q4_K_M
```

## Common operations

```bash
docker compose up -d                          # start full stack
docker compose up -d llama-server tailscale  # inference + tailnet only
docker compose logs -f llama-server
docker compose down
docker compose up -d --build llama-server    # rebuild after Dockerfile changes
docker compose restart llama-server openclaw-gateway  # after changing MODEL_NAME/QUANT
```

> After switching models, also update the `id` and `name` fields in `openclaw/openclaw.json`.

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List loaded model |
| `POST` | `/v1/chat/completions` | Chat completions |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/embeddings` | Embeddings |
| `POST` | `/slots/{id}?action=save` | Save slot KV cache to disk |
| `POST` | `/slots/{id}?action=restore` | Restore slot KV cache from disk |

## Troubleshooting

**Model file not found** — run the prepare step first:
```bash
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0
```

**TurboQuant: no fp16/bf16 source** — use `convert-st` instead of `download`.

**Out of VRAM** — reduce context or slots in `.env`:
```bash
LLAMA_CTX_SIZE=65536
LLAMA_PARALLEL=1
```

**Cloudflare Tunnel not connecting** — check `docker compose logs -f cloudflared` and confirm `CF_TUNNEL_TOKEN` is set and the Public Hostname points to `http://open-webui:8080`.

**Tailscale not joining** — check `docker compose logs -f tailscale`. Auth keys are single-use; generate a new one if the container restarted after first auth.

**Open WebUI shows pipeline connection error** — the pipelines server waits on llama-server. Wait for the model to finish loading (`docker compose logs llama-server`) then reload.

**GPU not visible** — verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`. See [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## License

MIT. llama.cpp: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp). OpenClaw: [openclaw/openclaw](https://github.com/openclaw/openclaw).
