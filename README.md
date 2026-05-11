# Custom LLM Server with llama.cpp & TurboQuant

A Docker-based host for running custom local LLMs using the latest llama.cpp with TurboQuant support.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- NVIDIA Container Toolkit installed (for GPU support)
- At least 16GB RAM for Qwopus3.6-35B-A3B model (35B params × 2 bytes for FP16 = 70GB if unquantized)

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd custom-llama
cp .env.default .env
```

### 2. One-Command Build and Run

#### GPU with model download and TurboQuant conversion:

```bash
MODEL_NAME=qwopus3.6-35b MODEL_QUANT=Q8_0 TQ_QUANT=TQ2_0 docker compose up -d
```

#### GPU with existing model:

```bash
docker compose up -d
```

#### CPU with model download:

```bash
MODEL_NAME=llama3.1-8b MODEL_QUANT=Q4_K_M docker compose -f docker-compose.cpu.yml up -d
```

### 3. Test the Server

```bash
curl http://localhost:8080/health
```

### 4. Send a Request

```bash
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

## Features

- **Docker containers** with llama.cpp from TheTom/llama-cpp-turboquant fork for full TurboQuant KV-cache support
- **NVIDIA GPU acceleration** via CUDA
- **OpenAI-compatible API** for easy integration
- **TurboQuant quantization** (TQ1_0, TQ2_0) - 1-bit/2-bit extreme compression
- **GGML_BLAS acceleration** for faster GEMM operations
- **One-command model download** - specify model and quantization at startup
- **Flexible configuration** via environment variables
- **Multimodal support** - Vision/image input for compatible models (e.g., Qwopus3.6-35B-A3B-v1)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_HOST` | `0.0.0.0` | Host to bind the server |
| `LLAMA_PORT` | `8080` | Port to bind the server |
| `LLAMA_MODEL` | `model.gguf` | Model filename in `/models` |
| `LLAMA_THREADS` | `8` | Number of CPU threads |
| `LLAMA_CTX_SIZE` | `4096` | Context window size |
| `LLAMA_GPU_LAYERS` | `99` | Layers to offload to GPU |
| `LLAMA_MAX_TOKENS` | `512` | Max generation length |
| `LLAMA_TOP_P` | `0.95` | Top-p sampling |
| `LLAMA_TEMP` | `0.7` | Sampling temperature |
| `LLAMA_STOP` | - | Stop sequences (comma-separated) |
| `HF_TOKEN` | - | HuggingFace token for gated models |
| `MODEL_NAME` | - | Model name to download from HuggingFace |
| `MODEL_QUANT` | `Q4_K_M` | Quantization for download |
| `TQ_QUANT` | - | TurboQuant quantization (TQ1_0, TQ2_0) |

### Available Models

| Model | Description |
|-------|-------------|
| `llama3.1-8b` | Meta Llama 3.1 8B Instruct (4-bit) |
| `llama3.1-70b` | Meta Llama 3.1 70B Instruct (4-bit) |
| `llama3.1-405b` | Meta Llama 3.1 405B Instruct (4-bit) |
| `phi3-mini` | Microsoft Phi-3 Mini 4K Instruct (4-bit) |
| `mistral-7b` | Mistral 7B Instruct (4-bit) |
| `mixtral-8x7b` | Mistral MoE 8x7B Instruct (4-bit) |
| `qwopus3.6-35b` | Qwopus 3.6 35B-A3B v1 (4-bit MoE, 3.1B active params) **[Multimodal]** |

### Multimodal Models

The **Qwopus3.6-35B-A3B-v1** model supports image input (multimodal). To use it:

1. Download the model and its multimodal projector file:
   ```bash
   MODEL_NAME=qwopus3.6-35b MODEL_QUANT=Q8_0 MMPROJ=/models/Qwopus3.6-35B-A3B-v1-Q8_0-mmproj.gguf docker compose up -d
   ```

2. The `mmproj.gguf` file will be automatically downloaded alongside the model.

3. Enable multimodal mode by setting the environment variables:
   ```bash
   LLAMA_MMPROJ=/models/mmproj.gguf
   ```

4. Start the server with multimodal support:
   ```bash
   docker compose up -d
   ```

See [Multimodal Image Input](#multimodal-image-input) below for details on sending images.

### GGUF Quantization Options

| Quantization | Description |
|--------------|-------------|
| `Q4_K_M` | **Recommended** - Best quality for size |
| `Q5_K_M` | Slightly better quality |
| `Q6_K` | High quality |
| `Q8_0` | 8-bit - best source for TurboQuant conversion |
| `Q4_0` | Basic 4-bit |
| `Q3_K_M` | Smaller size |
| `IQ4_XS` | Even smaller |

### TurboQuant (TQ1_0, TQ2_0)

TurboQuant is llama.cpp's 1-bit/2-bit quantization format for extreme compression. It provides:
- **TQ1_0**: ~1-bit per weight - extreme compression, suitable for very large models
- **TQ2_0**: ~2-bit per weight - better quality while still highly compressed

TurboQuant models use `.gguf` files with `TQ1_0` or `TQ2_0` quantization types. These are only suitable for models specifically converted to TurboQuant format (not all models are available in this format).

**Important: Converting from a higher-precision source produces better TurboQuant results.** TurboQuant is a lossy compression method - it compresses weights from the source model to ~1-bit or ~2-bit. The more information preserved in the source, the better the TurboQuant output.

#### Recommended Conversion Sources

| Source Model | Target | Quality | Approximate Size |
|-------------|--------|---------|-----------------|
| `FP16` | TQ2_0 | **Best** | ~8-9 GB |
| `FP16` | TQ1_0 | **Best** | ~4-5 GB |
| `Q8_0` | TQ2_0 | **Best** | ~8-9 GB |
| `Q6_K` | TQ2_0 | Good | ~7-8 GB |
| `Q5_K_M` | TQ2_0 | Acceptable | ~6-7 GB |
| `Q4_K_M` | TQ2_0 | Noticeable loss | ~5-6 GB |

For the best TurboQuant quality, **always convert from FP16 if available**. If FP16 is not available on HuggingFace, **always convert from Q8_0**.

#### Converting to TurboQuant

You can convert **any GGUF model** to TurboQuant format, including FP16 if available:

```bash
# Download a model and convert to TurboQuant in one command
MODEL_NAME=qwopus3.6-35b MODEL_QUANT=Q8_0 TQ_QUANT=TQ2_0 docker compose up -d

# Or use the model manager directly
docker compose run --rm model-manager turboquant /models/Qwopus3.6-35B-A3B-v1-Q8_0.gguf -q TQ2_0
```

## Multimodal Image Input

For multimodal models like Qwopus3.6-35B-A3B-v1, you can send images along with text prompts. The model uses a multimodal projector (`mmproj.gguf`) to encode images into the same embedding space as text.

### Sending Images via API

Use the `images` field in the message content to include images:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."}}
      ]}
    ]
  }'
```

### Base64-encoded Images

You can also send images as base64-encoded data:

```bash
# First, encode the image to base64
base64 -w 0 image.png > image_b64.txt

# Then send the request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'$(cat image_b64.txt)'"}}
      ]}
    ]
  }'
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MMPROJ` | - | Path to the multimodal projector file (.mmproj) |
| `LLAMA_IMAGE` | - | Path to a local image file for initial input |

### Supported Image Formats

- PNG
- JPEG
- BMP
- GIF

### Notes

- The `mmproj.gguf` file must be placed in the same directory as the model file
- The model must be a multimodal-compatible model (e.g., Qwopus3.6-35B-A3B-v1, LLaVA variants)
- Image resolution is automatically handled by the model's vision encoder

## Model Management

### Download a Model

```bash
# From HuggingFace (requires huggingface-cli)
docker compose run --rm model-manager download llama3.1-8b -q Q4_K_M

# Convert a model to GGUF format
docker compose run --rm model-manager convert /path/to/model.gguf -q Q4_K_M

# Convert to TurboQuant
docker compose run --rm model-manager turboquant /models/model.gguf -q TQ2_0
```

### List Available Models

```bash
docker compose run --rm model-manager list
```

## API Endpoints

### Chat Completions (OpenAI-compatible)

```
POST /v1/chat/completions
```

### Completions

```
POST /v1/completions
```

### Health Check

```
GET /health
```

### Model Info

```
GET /models
```

## GPU Setup

### Install NVIDIA Container Toolkit

```bash
# For Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

## Docker Compose Commands

```bash
# Build and start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f llama-server

# Rebuild
docker compose up -d --build

# With GPU
docker compose up -d

# CPU only (remove GPU runtime)
docker compose -f docker-compose.cpu.yml up -d
```

## Custom Models

Place your GGUF model files in the `models/` directory. The model will be mounted read-only into the container.

```bash
# Download a model manually
mkdir -p models
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models
```

Then set `LLAMA_MODEL` in your `.env` file to your model filename.

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Client                      │
│  (OpenAI-compatible API calls)               │
└──────────────────┬──────────────────────────┘
                   │ HTTP/8080
┌──────────────────▼──────────────────────────┐
│           llama-server (Container)           │
│  ┌─────────────────────────────────────────┐ │
│  │  llama.cpp (TurboQuant KV-cache fork)    │ │
│  │  - CUDA GPU acceleration                 │ │
│  │  - OpenAI-compatible API                 │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              models/ (Volume)                │
│  - model.gguf (GGUF format)                  │
│  - TurboQuant quantized models               │
│  - mmproj.gguf (multimodal)                  │
└─────────────────────────────────────────────┘
```

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA Container Toolkit
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Model not found

```bash
# Check mounted volume
docker exec llama-server ls -la /models/

# Check model file permissions
ls -la models/
```

### Out of memory

```bash
# Reduce GPU layers
LLAMA_GPU_LAYERS=30 docker compose up -d

# Reduce context size
LLAMA_CTX_SIZE=2048 docker compose up -d
```

## License

This project is provided as-is. llama.cpp is licensed under MIT (see [llama.cpp](https://github.com/ggerganov/llama.cpp)).
