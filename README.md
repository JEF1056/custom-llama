# custom-llama

A self-hosted LLM inference server built around [llama.cpp (TurboQuant fork)](https://github.com/TheTom/llama-cpp-turboquant). Exposed publicly via Cloudflare Tunnel with Cloudflare Access authentication.

## Local quick start

No Cloudflare, or secrets needed — just the inference server on this machine.

```bash
cp .env.default .env   # set MODEL_NAME + QUANT; leave everything else blank
docker compose build && docker compose build llama-convert
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant Q3_K_L
docker compose up -d llama-server
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

---

## Architecture

```
              ┌──────────────────────────────────┐
              │         Cloudflare Edge           │
              │  api.jessfan.com                  │
              └──────────────┬────────────────────┘
                             │ Cloudflare Tunnel (outbound)
                             │ Cloudflare Access (auth required)
              ┌──────────────▼────────────────────┐
              │         Host Machine               │
              │  cloudflared → llama-server :8080  │
              └───────────────┬────────────────────┘
                                │ llama-net (internal)
                          llama-server :8080
```

| Interface | URL / Command | Auth |
|---|---|---|
| **Local** | `http://localhost:8080/v1` | None (requires `docker-compose.override.yml`) |
| **API (Cloudflare Access)** | `https://api.jessfan.com/v1` | Google OAuth / Email (see below) |

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
cp .env.default .env
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

### Step 4: Build and prepare a model

```bash
# Build the containers
docker compose build
docker compose build llama-convert

# Download and quantize a model (choose one):
# Pre-built GGUF on HuggingFace
docker compose run --rm llama-convert download qwen3.5-27b --quant Q4_K_M

# OR Safetensors-only repos (convert → fp16 → quantize)
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant Q3_K_L
```

> **Gated models:** set `HF_TOKEN=your_token` in `.env`

### Step 5: Start the services

```bash
docker compose up -d
```

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
     -d '{"model": "qwen3.5-27b-Q4_K_M", "messages": [{"role": "user", "content": "Hello"}]}'
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
    model="qwen3.5-27b-Q4_K_M",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Docker Compose services

| Service | Purpose |
|---|---|
| `llama-server` | llama.cpp inference server (port 8080) |
| `cloudflared` | Cloudflare Tunnel — exposes llama-server publicly |
| `llama-convert` | Model conversion tool (download, convert, quantize) |

## Troubleshooting

- **Model not loading:** Check `docker compose logs llama-server` for errors. Common issues: model path mismatch, insufficient VRAM.
- **Cloudflare Tunnel not connecting:** Verify `CF_TUNNEL_TOKEN` is correct. Check `docker compose logs cloudflared`.
- **Cloudflare Access authentication failing:** Ensure the Access Application is configured for the correct domain and authentication method.
- **GPU not detected:** Verify NVIDIA Container Toolkit is installed and working. Check `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`.
