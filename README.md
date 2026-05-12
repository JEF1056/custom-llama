# custom-llama

A Docker-based llama.cpp server with full [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) KV-cache support, OpenAI-compatible API, and a dedicated model-preparation image that handles downloading, quantization, and safetensors conversion.

## Architecture

Two Docker images share a `./models` volume:

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐
│        llama-convert            │     │          llama-server            │
│  (model prep — run once)        │     │  (server — run continuously)     │
│                                 │     │                                  │
│  • download pre-built GGUFs     │     │  • llama-server binary only      │
│  • quantize from fp16 GGUF      │ ──▶ │  • reads model from /models      │
│  • convert safetensors → GGUF   │     │  • CUDA GPU acceleration         │
│  • re-quantize existing GGUFs   │     │  • OpenAI-compatible API         │
└──────────────┬──────────────────┘     └──────────────────────────────────┘
               │ writes
       ┌───────▼────────┐
       │  ./models/     │
       │  (bind mount)  │
       └────────────────┘
```

## Prerequisites

- Docker + Docker Compose
- NVIDIA GPU (RTX 30xx or newer recommended for TurboQuant KV-cache)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/JEF1056/custom-llama.git
cd custom-llama
cp .env.default .env
# Edit .env — set MODEL_NAME, QUANT, and any server parameters
```

### 2. Build both images

```bash
docker compose build
docker compose build llama-convert  # builds the convert image (profile: convert)
```

### 3. Prepare a model

All model preparation is done via the convert image. The output lands in `./models/`.

```bash
# Download a pre-built GGUF (most models)
docker compose run --rm llama-convert download qwen3.5-27b --quant Q4_K_M

# Download and quantize locally (no pre-built quant on HuggingFace)
docker compose run --rm llama-convert download qwen3.6-27b --quant TQ2_0

# Safetensors-only repo: convert to fp16 GGUF first, then quantize
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0

# List all available models
docker compose run --rm llama-convert list
```

> **Gated / private models:** set `HF_TOKEN=your_token` in `.env` before running
> the convert image. The token is passed through automatically.

### 4. Start the server

```bash
docker compose up -d
```

The server reads `MODEL_NAME` and `QUANT` from `.env`, constructs the path
`/models/{MODEL_NAME}-{QUANT}.gguf`, and starts immediately. If the file is
missing it exits with a helpful error showing the exact `docker compose run`
command to fix it.

### 5. Test

```bash
curl http://localhost:8080/health

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

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

Downloaded directly from HuggingFace when available; otherwise the best
available source GGUF (fp16 → bf16 → Q8_0 → …) is downloaded and quantized
locally with `llama-quantize`.

| Quant | Size vs Q4_K_M | Notes |
|-------|----------------|-------|
| `Q4_K_M` | 1× | Recommended default — best quality/size balance |
| `Q5_K_M` | 1.2× | Slightly higher quality |
| `Q6_K` | 1.5× | High quality |
| `Q8_0` | 2× | Near-lossless; good intermediate for re-quantization |
| `Q3_K_M` | 0.75× | Smaller, some quality loss |
| `IQ4_XS` | 0.9× | Imatrix-optimized 4-bit |

### TurboQuant (TQ2_0 / TQ1_0)

TurboQuant is llama.cpp's extreme compression format (~2-bit and ~1-bit per
weight). It requires **fp16 or bf16 GGUF** as the quantization source —
lower quants such as Q8_0 are not accepted.

| Quant | Bits/weight | Notes |
|-------|-------------|-------|
| `TQ2_0` | ~2 | Better quality, ~4–5× smaller than fp16 |
| `TQ1_0` | ~1 | Maximum compression, some quality loss |

**Source availability by case:**

```
fp16/bf16 GGUF in hf_repo  →  download command handles it automatically
No fp16 GGUF, safetensors only  →  use convert-st (convert image)
```

```bash
# Case 1: fp16 GGUF exists on HuggingFace (auto-download + quantize)
docker compose run --rm llama-convert download qwen3.6-27b --quant TQ2_0

# Case 2: safetensors only (downloads safetensors, converts, quantizes)
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0
```

## Configuration

Copy `.env.default` to `.env` and adjust. All variables have sensible defaults
for an RTX 3090 (24 GB VRAM) desktop.

### Model selection

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `qwopus3.6-35b` | Key from the models list |
| `QUANT` | `TQ2_0` | Quantization type |
| `LLAMA_MODEL` | _(empty)_ | Override: explicit path to a `.gguf` in `/models` |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_HOST` | `0.0.0.0` | Bind address |
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
| `LLAMA_STOP` | _(empty)_ | Stop sequences (space-separated) |
| `LLAMA_NO_MMAP` | `off` | `on` = load model into RAM; `off` = mmap (default) |

### KV cache

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CACHE_TYPE_K` | `turbo3` | K-cache type: `f16`, `turbo3`, `turbo4` |
| `LLAMA_CACHE_TYPE_V` | `turbo3` | V-cache type: `f16`, `turbo3`, `turbo4` |
| `LLAMA_FLASH_ATTN` | `on` | Flash Attention: `on`, `off`, or `auto`. Required for TurboQuant KV cache |
| `LLAMA_NO_KV_OFFLOAD` | `off` | `on` = keep KV cache on GPU at all times |

### Reasoning / chat

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_REASONING` | `on` | Chain-of-thought output: `on`, `off`, or `auto` (detect from template) |
| `LLAMA_PRESERVE_THINKING` | `on` | Include prior `<think>` blocks in context |

### Security / persistence

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_API_KEY` | _(empty)_ | Bearer token required on all requests. Leave empty for open access |
| `LLAMA_SLOT_SAVE_PATH` | _(empty)_ | Directory to persist KV-cache slot state across restarts (`/slots/{id}?action=save\|restore`) |

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

## Multimodal (Vision)

Models tagged **[Multimodal]** automatically download their `mmproj.gguf` alongside
the main model. Set `LLAMA_MMPROJ` in `.env` to the downloaded projector path before
starting the server.

```bash
# Prepare model + mmproj
docker compose run --rm llama-convert download qwopus3.6-35b --quant TQ2_0
# mmproj.gguf is downloaded automatically into ./models/

# In .env:
# LLAMA_MMPROJ=/models/mmproj.gguf

docker compose up -d
```

Send an image via the API:

```bash
curl http://localhost:8080/v1/chat/completions \
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

## Model Management Commands

All commands run via `docker compose run --rm llama-convert <command>`.

```bash
# List all supported models with sizes
docker compose run --rm llama-convert list

# Download a pre-built GGUF (or quantize locally if quant not on HF)
docker compose run --rm llama-convert download <model> --quant <quant>

# Convert safetensors → fp16 GGUF → target quant (for TQ2_0 when no GGUF source exists)
docker compose run --rm llama-convert convert-st <model> --quant <quant>

# Re-quantize an existing GGUF already in ./models
docker compose run --rm llama-convert convert /models/model-Q8_0.gguf --quant Q4_K_M

# Convert an existing GGUF to TurboQuant
docker compose run --rm llama-convert turboquant /models/model-fp16.gguf --quant TQ2_0
```

## API Reference

The server exposes an OpenAI-compatible HTTP API on `LLAMA_PORT` (default `8080`).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List loaded model |
| `POST` | `/v1/chat/completions` | Chat completions |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/embeddings` | Embeddings |

## Common Operations

```bash
# Start server (detached)
docker compose up -d

# View server logs
docker compose logs -f llama-server

# Stop server
docker compose down

# Rebuild server image
docker compose up -d --build

# Switch models: edit MODEL_NAME/QUANT in .env, then restart
docker compose restart llama-server
```

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

The GGUF repo only has lower-precision files. Use `convert-st` to download from
the safetensors repo and convert:

```bash
docker compose run --rm llama-convert convert-st <model> --quant TQ2_0
```

### Out of VRAM

Reduce context or parallel slots in `.env`:

```bash
LLAMA_CTX_SIZE=65536
LLAMA_PARALLEL=1
```

Or offload fewer layers:

```bash
LLAMA_GPU_LAYERS=40
```

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

This project is provided as-is. llama.cpp is licensed under MIT — see [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).
