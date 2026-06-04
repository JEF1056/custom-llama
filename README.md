# custom-llama

A self-hosted LLM inference server powered by [vLLM](https://github.com/vllm-project/vllm) v0.22.0. Serves AutoRound INT4 safetensors models with TurboQuant KV cache compression, MTP speculative decoding, and vision support. Exposed publicly via Cloudflare Tunnel with Cloudflare Access authentication.

**Default model:** [Qwen3.6-27B AutoRound INT4](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — 19 GB, native MTP heads, vision support, 128K context on a single RTX 3090.

---

## Local quick start

No Cloudflare or secrets needed — just the inference server on this machine.

```bash
# 1. Generate .env from defaults (auto-generates LLM_API_KEY, MCP_API_KEY)
python3 sync-env.py

# 2. Build images
docker compose build vllm-server model-prep mcp-search-server

# 3. Download the default AutoRound INT4 model (~19 GB)
docker compose run --rm model-prep download qwen3.6-27b-autoround

# 4. .env already defaults to this model — no changes needed
#    LLM_MODEL_PATH=/models/qwen3.6-27b-autoround
#    LLM_SERVED_MODEL_NAME=qwen3.6-27b

# 5. Start
docker compose up -d vllm-server mcp-search-server
```

Port 8080 is exposed via `docker-compose.override.yml` (gitignored) for local dev. Test with:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
```

Any OpenAI-compatible client (Cursor, Roo Code, opencode, etc.) points at `http://localhost:8080/v1`.

### Local port overrides (`docker-compose.override.yml`)

Create this file at the project root to expose all services to your local machine. It is gitignored — **do not commit it**.

```yaml
# docker-compose.override.yml
# Exposes all services to localhost for local development.
# gitignored — do not commit.
services:
  vllm-server:
    restart: "no"
    ports:
      - "8080:8080"
    networks:
      - llama-net
      - host-bridge

  mcp-search-server:
    ports:
      - "3100:3100"
    networks:
      - llama-net
      - host-bridge

  chat-ui:
    ports:
      - "5173:5173"
    networks:
      - llama-net
      - host-bridge

networks:
  host-bridge:
    driver: bridge
```

| Service             | Local URL               |
| ------------------- | ----------------------- |
| `vllm-server`       | `http://localhost:8080` |
| `mcp-search-server` | `http://localhost:3100` |
| `chat-ui`           | `http://localhost:5173` |

> **Note:** `vllm-server` intentionally has no port mapping in `docker-compose.yml` — it is only reachable via Cloudflare Tunnel in production. The override adds the localhost binding for local dev only.

---

## Images

| Image               | Dockerfile                     | Purpose                                                   |
| ------------------- | ------------------------------ | --------------------------------------------------------- |
| `vllm-server`       | `Dockerfile`                   | vLLM v0.22.0 inference server (official or fork)          |
| `model-prep`        | `Dockerfile.modelprep`         | Lightweight Python image for downloading AutoRound models |
| `mcp-search-server` | `mcp-search-server/Dockerfile` | Web search MCP tool                                       |

### vLLM server (`Dockerfile`)

Uses the official `vllm/vllm-openai:v0.22.0` image with a custom entrypoint. Key features:

- **AutoRound INT4** — pre-quantized safetensors, auto-detected by vLLM from `quantize_config.json`. Fits 19 GB on RTX 3090, leaving ~3.5 GB raw for KV cache.
- **TurboQuant KV cache** — `turboquant_k4v2_nc` preset (4-bit keys + 2-bit values + norm correction, 5.0× KV compression). Keys get more bits because they're ~37× more sensitive to quantization than values. Requires [vllm-turboquant fork](https://github.com/JEF1056/vllm-turboquant/tree/turboquant-k4v2-nc).
- **MTP speculative decoding** — uses the model's native multi-token prediction heads via `--speculative-config`. No separate draft model, zero extra VRAM, ~70-85% acceptance rate.
- **Reasoning parser** — structured `<think>…</think>` extraction via `--reasoning-parser qwen3`.
- **Vision** — auto-detected from Qwen3.6 checkpoint, no flag needed.

#### Building from a vLLM fork

Set `VLLM_FORK_REPO` and `VLLM_FORK_BRANCH` to overlay Python source from a fork onto the official image. Compiled C/CUDA extensions are preserved — only Python and Triton kernels (JIT-compiled at runtime) are replaced. Build time: ~30 seconds.

Use this for custom TurboQuant presets, attention backends, model definitions, or scheduling logic. For C++/CUDA kernel changes, use vLLM's upstream `docker/Dockerfile` instead (30-60 min full build).

```bash
# Build from a fork (overlays Python source, ~30s)
VLLM_FORK_REPO=https://github.com/you/vllm.git VLLM_FORK_BRANCH=feat/k4v2 docker compose build vllm-server

# Build from the official image (default when vars are unset)
docker compose build vllm-server
```

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
              │  cloudflared → vllm-server:8080   │
              └───────────────┬────────────────────┘
                              │ llama-net (internal)
                    ┌─────────────────────────┐
                    │  vllm-server :8080      │
                    │  mcp-search-server :3100│
                    └─────────────────────────┘
```

| Interface  | URL                           | Auth                                          |
| ---------- | ----------------------------- | --------------------------------------------- |
| **Local**  | `http://localhost:8080/v1`    | None (requires `docker-compose.override.yml`) |
| **Public** | `https://chat.jessfan.com/v1` | Cloudflare Access (Google OAuth / Email)      |

---

## Step-by-step setup guide

### Step 1: Create Cloudflare Tunnel

1. Go to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/) → **Networks → Tunnels**
2. Click **Create a tunnel** → **Docker** → copy the token
3. Paste into `.env` as `CF_TUNNEL_TOKEN`
4. Add Public Hostname: `chat.jessfan.com` → `http://vllm-server:8080`

### Step 2: Configure `.env`

```bash
python3 sync-env.py
```

Edit `.env` and set at minimum:

```bash
# Cloudflare Tunnel token
CF_TUNNEL_TOKEN=eyJhIjoi...

# Model defaults are already correct after downloading:
LLM_MODEL_PATH=/models/qwen3.6-27b-autoround
LLM_SERVED_MODEL_NAME=qwen3.6-27b
```

`LLM_API_KEY` and `MCP_API_KEY` are auto-generated by `sync-env.py` if empty.

### Step 3: Download a model

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

### Step 4: Build and start

```bash
# Build vLLM server (uses official vllm-openai image — fast)
docker compose build vllm-server

# Start inference server + MCP search tool
docker compose up -d vllm-server mcp-search-server

# With Cloudflare Tunnel
docker compose up -d
```

Check logs:

```bash
docker compose logs -f vllm-server     # allow up to 5 min for large model load
docker compose logs -f cloudflared     # should show "connected"
```

### Step 5: Test

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/models

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Step 6: Connect a client

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

| Variable                     | Default                                       | Description                                                      |
| ---------------------------- | --------------------------------------------- | ---------------------------------------------------------------- |
| `LLM_MODEL_PATH`             | `/models/qwen3.6-27b-autoround`               | Local safetensors directory inside container                     |
| `LLM_QUANTIZATION`           | —                                             | Auto-detected from checkpoint; override with `gptq`, `awq`, etc. |
| `LLM_TOKENIZER_PATH`         | —                                             | Leave empty — bundled in model repo                              |
| `LLM_SERVED_MODEL_NAME`      | `qwen3.6-27b`                                 | Model alias for `/v1/models`                                     |
| `LLM_MAX_MODEL_LEN`          | `128000`                                      | Max context window in tokens                                     |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.95`                                        | GPU VRAM fraction for weights + KV                               |
| `LLM_MAX_NUM_SEQS`           | `1`                                           | Concurrent request slots (1 for 128K context)                    |
| `LLM_TP_SIZE`                | `1`                                           | Tensor parallelism (number of GPUs)                              |
| `LLM_KV_CACHE_DTYPE`         | `turboquant_k4v2_nc`                          | KV cache quantization (5.0× compression)                         |
| `LLM_REASONING_PARSER`       | `qwen3`                                       | Reasoning extraction parser                                      |
| `LLM_SPECULATIVE_CONFIG`     | `{"method":"mtp","num_speculative_tokens":1}` | MTP speculative decoding config                                  |
| `LLM_ENFORCE_EAGER`          | —                                             | Set to `1` to save ~1.5 GB VRAM (slower decode)                  |
| `LLM_API_KEY`                | auto-generated                                | Bearer token for API auth                                        |

### RTX 3090 VRAM budget

Qwen3.6-27B is a **hybrid model**: 64 layers total, but only **16 full-attention layers** cache KV (the other 48 are linear-attention with fixed-size recurrent state). This dramatically reduces KV cache cost vs. a pure transformer.

| Component                   | VRAM            | Notes                                  |
| --------------------------- | --------------- | -------------------------------------- |
| INT4 weights + scales       | ~15.5 GB        | 27B × 0.5 B/param + 15% group overhead |
| Embeddings + LM head (BF16) | ~2.4 GB         | vocab 248K × hidden 5120 × 2 × 2B      |
| Vision tower (BF16)         | ~0.4 GB         | 27-layer ViT encoder                   |
| MTP heads (BF16)            | ~0.2 GB         | Native multi-token prediction          |
| Linear attn state (fixed)   | ~0.1 GB         | 48 layers, does not grow with context  |
| CUDA context + activations  | ~0.5 GB         |                                        |
| **Total non-KV**            | **~19.1 GB**    |                                        |
| **KV cache budget**         | **~3.7 GB raw** | 22.8 GB usable − 19.1 GB               |

**KV cache per token:** 2 × 16 layers × 4 KV heads × 256 dim × 2B = **64 KB**

| KV dtype               | Compression | Token capacity | Max context (×1 seq) |
| ---------------------- | ----------- | -------------- | -------------------- |
| BF16 (auto)            | 1×          | ~59K           | 59K                  |
| fp8                    | 2×          | ~118K          | 118K                 |
| turboquant_k8v4        | 2.6×        | ~154K          | 154K                 |
| turboquant_4bit_nc     | 3.8×        | ~225K          | 225K                 |
| **turboquant_k4v2_nc** | **5.0×**    | **~295K**      | **295K**             |

128K fits with ~167K tokens of headroom.

---

## Docker Compose services

| Service             | Purpose                                                         |
| ------------------- | --------------------------------------------------------------- |
| `vllm-server`       | vLLM inference server (port 8080)                               |
| `cloudflared`       | Cloudflare Tunnel — exposes vllm-server publicly                |
| `model-prep`        | Download AutoRound models into `./models/` — profile: `convert` |
| `mcp-search-server` | Web search MCP tool (port 3100)                                 |
| `chat-ui`           | Vite + React chat interface (port 5173)                         |

---

## Benchmarking

A phased benchmarking framework measures tokens/second across different server configurations to find the optimal `.env.default` settings.

### Quick start

```bash
pip install httpx
python scripts/benchmark.py
```

This runs all 4 phases (~2.75 hours total) and generates a Markdown report.

### Phases (most → least impactful)

| Phase                    | What it sweeps                        | Restarts | Est. time |
| ------------------------ | ------------------------------------- | -------- | --------- |
| 1 — Speculative decoding | none / MTP / ngram / ngram+MTP        | 4        | ~55 min   |
| 2 — KV cache dtype       | turboquant (k4v2, k3v2, 3bit)         | 3        | ~55 min   |
| 3 — DRY sampling         | on vs off (per-request toggle)        | 0-1      | ~15 min   |
| 4 — Cross-validation     | top 3 configs, 5 runs each            | ≤3       | ~40 min   |

Each phase isolates one variable. The winner carries forward as the baseline for the next phase.

### Options

```bash
python scripts/benchmark.py --phase 1       # run only Phase 1
python scripts/benchmark.py --runs 5        # 5 runs per scenario (default 3)
python scripts/benchmark.py --resume        # resume an interrupted run
python scripts/benchmark.py --report-only   # regenerate report from existing data
python scripts/benchmark.py --stop-on-error # abort on first error instead of continuing
```

### Output

- `benchmark/results/{timestamp}_runs.jsonl` — raw data, one JSON object per run (flushed immediately)
- `benchmark/results/{timestamp}_report.md` — Markdown report with tables, % comparisons, and a recommended `.env.default` config block

### How it works

1. For each config, the script sets env vars and runs `docker compose -f docker-compose.yml -f docker-compose.benchmark.yml up -d --force-recreate vllm-server` (no `.env` modification).
2. Waits for `/health` (up to 7 min for model load + compilation).
3. Runs 5 scenarios (general text, coding, agentic, instruction following, tool calling) × N runs each.
4. Measures TTFT and decode tokens/second via streaming API with `stream_options: {"include_usage": true}`.
5. Progress bar with ETA updates after every run.

---

## Troubleshooting

- **Model not loading:** Check `docker compose logs vllm-server`. Common causes: `LLM_MODEL_PATH` wrong, model directory empty, insufficient VRAM.
- **OOM at startup:** Lower `LLM_GPU_MEMORY_UTILIZATION` (try `0.92`) or reduce `LLM_MAX_MODEL_LEN`. If OOM occurs during CUDA graph capture, set `LLM_ENFORCE_EAGER=1`.
- **OOM at 128K:** Set `LLM_ENFORCE_EAGER=1` to reclaim ~1.5 GB, and ensure `LLM_MAX_NUM_SEQS=1`. The hybrid arch (only 16/64 layers cache KV) means 128K should fit without eager mode, but activation spikes can still cause OOM.
- **MTP not working:** Verify the model checkpoint includes MTP heads. Check `LLM_SPECULATIVE_CONFIG` is valid JSON. If acceptance rate is near 0%, try `"num_speculative_tokens": 2`.
- **TurboQuant error:** Ensure vLLM ≥ 0.20 (TurboQuant merged upstream April 2026). MLA models are not supported — TurboQuant raises `NotImplementedError` on MLA architectures.
- **Cloudflare Tunnel not connecting:** Verify `CF_TUNNEL_TOKEN`. Check `docker compose logs cloudflared`.
- **GPU not detected:** Verify NVIDIA Container Toolkit. Run `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- **Slow decode speed:** If `LLM_ENFORCE_EAGER=1`, you lose 10-30% decode speed. Unset it and set `VLLM_MAX_SEQ_LEN_TO_CAPTURE=8192` to re-enable CUDA graphs with bounded capture memory.
