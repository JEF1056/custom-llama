# Qwen3.6-35B-A3B MLX Deployment (macOS)

One-line deployment of the Qwen3.6-35B-A3B VLM server on Apple Silicon MacBooks.

## Quick Deploy

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | bash
```

Or with an authenticated token (optional for public repos, required for gated ones):

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | HF_TOKEN=hf_xxx bash
```

## What It Does

1. **Checks prerequisites** — macOS, Apple Silicon, Xcode CLT, Python 3
2. **Creates a venv** at `~/.qwen/mlx-venv` with `mlx-kquant`, `mlx-lm[torch]`, `mlx-vlm[torch]`
3. **Downloads the BF16 GGUF** (~71 GB) from HuggingFace
4. **Converts to MLX FP16 safetensors** using `mlx_lm.convert`
5. **Applies K-quant quantization** (Docker 262K-Balanced policy) → `~/.qwen/models/qwen36-mlx/quantized/` (~16-17 GB)
6. **Installs a LaunchAgent** (`com.custom-llama.qwen36-mlx`) that starts the server at login and auto-restarts on crash
7. **Starts the server** on port 8081

## Configuration

Override defaults via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace read token for model download |
| `MLX_PORT` | `8081` | Server port |
| `MLX_KV_BITS` | `4` | KV cache quantization bits |
| `MODEL_PATH` | `~/.qwen/models/qwen36-mlx/quantized` | Custom model path |
| `HF_REPO` | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF` | HF repo for GGUF |
| `HF_GGUF_FILE` | `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-bf16.gguf` | GGUF filename |
| `QWEN_HOME` | `~/.qwen` | Base directory for all qwen36 artifacts |
| `CUSTOM_LLAMA_REPO` | `https://github.com/YOURUSER/custom-llama.git` | Repo to fetch scripts from |
| `CUSTOM_LLAMA_REF` | `main` | Branch/commit to use |

### Example with custom settings

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | HF_TOKEN=hf_xxx QWEN_HOME=/Volumes/External/qwen MLX_PORT=8082 bash
```

## Server Usage

After install, the server is available at `http://localhost:8081/v1`.

### Text completion

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 100,
    "temperature": 0.6,
    "top_p": 0.95
  }'
```

### Vision (image input)

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "input_image", "image_url": "https://example.com/image.jpg"},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 200
  }'
```

### Thinking mode

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "Solve: 2+2"}],
    "max_tokens": 100,
    "enable_thinking": true
  }'
```

## Fleet Deployment

For deploying to multiple MacBooks, use the same `install.sh` with each machine's IP:

```bash
# On each MacBook:
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | HF_TOKEN=hf_xxx bash

# Then add all Macs to router/config.yaml:
#   - model_name: qwen3.6-35b
#     litellm_params:
#       model: openai/qwen3.6-35b
#       api_base: http://<m5-pro-ip>:8081/v1
#   - model_name: qwen3.6-35b
#     litellm_params:
#       model: openai/qwen3.6-35b
#       api_base: http://<m4-pro-ip>:8081/v1
#   - model_name: qwen3.6-35b
#     litellm_params:
#       model: openai/qwen3.6-35b
#       api_base: http://<m1-pro-ip>:8081/v1
```

The LiteLLM router uses latency-based routing, automatically preferring faster Macs.

## Uninstall

```bash
bash ~/.qwen/custom-llama/mac/uninstall.sh
```

This removes the LaunchAgent but leaves downloaded models in `~/.qwen` to reclaim disk space manually.

## Directory Structure

After install:

```
~/.qwen/
├── mlx-venv/              # Python virtual environment
├── models/
│   ├── Qwen3.6-35B-*.bf16.gguf   # Downloaded BF16 GGUF (~71 GB)
│   └── qwen36-mlx/            # FP16 MLX safetensors
│       ├── config.json
│       ├── model*.safetensors
│       └── tokenizer*
│   └── qwen36-mlx/quantized/  # K-quant quantized model (~16-17 GB)
│       ├── config.json
│       ├── model.safetensors
│       ├── kquant_preset.json
│       └── tokenizer*
├── custom-llama/          # Cloned repo (scripts/, mac/)
└── Library/LaunchAgents/
    └── com.custom-llama.qwen36-mlx.plist
```

## Logs

- **Server stdout:** `~/Library/Logs/qwen36-mlx.out.log`
- **Server stderr:** `~/Library/Logs/qwen36-mlx.err.log`

## Troubleshooting

### Server won't start at login

Check the LaunchAgent log:
```bash
tail -f ~/Library/Logs/qwen36-mlx.err.log
```

Common issues:
- `ModuleNotFoundError: No module named 'mlx_vlm'` — venv PATH not set; re-run install.sh
- `Model directory not found` — quantization step failed; check `~/.qwen/models/qwen36-mlx/quantized/`

### Re-run install.sh

The script is idempotent — it skips steps where artifacts already exist. To force a fresh install:

```bash
# Remove existing artifacts
rm -rf ~/.qwen/models/qwen36-mlx ~/.qwen/models/qwen36-mlx/quantized
# Re-run
curl -fsSL https://raw.githubusercontent.com/YOURUSER/custom-llama/main/mac/install.sh \
  | HF_TOKEN=hf_xxx bash
```
