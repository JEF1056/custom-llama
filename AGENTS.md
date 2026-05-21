# custom-llama — Agent Instructions

## Commands

### Model conversion (the `llama-convert` image — CPU-only)

```bash
# Pre-built GGUF on HuggingFace:
docker compose run --rm llama-convert download qwen3.5-27b --quant Q4_K_M

# fp16 GGUF exists but no quant (quantize locally):
docker compose run --rm llama-convert download qwen3.6-27b --quant TQ2_0

# Safetensors-only repo (convert → fp16 GGUF → quantize):
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant TQ2_0

# MTP-capable GGUF (auto-grafts MTP head from base model):
docker compose run --rm llama-convert convert-st qwopus3.6-27b --quant Q3_K_M --mtp
docker compose run --rm llama-convert convert-st qwopus3.6-35b --quant Q3_K_L --mtp
# ↑ Both auto-graft MTP from their respective base models:
#   qwopus3.6-27b ← Qwen/Qwen3.6-27B    (15 MTP tensors, dense)
#   qwopus3.6-35b ← Qwen/Qwen3.6-35B-A3B (19 MTP tensors, MoE)
# Then set in .env: LLAMA_MODEL=/models/{model}-Q3_K_L-mtp.gguf
#                   LLAMA_SPEC_TYPE=mtp  LLAMA_SPEC_DRAFT_N_MAX=3

# Re-quantize an existing GGUF already in ./models:
docker compose run --rm llama-convert convert /models/model-Q8_0.gguf --quant Q4_K_M
```

### Server

```bash
# Build both images (only if Dockerfile changed):
docker compose build && docker compose build llama-convert

# Start all services:
# Note: llama-convert needs --profile convert, cloudflared needs --profile cloudflare.
docker compose up -d

# Logs:
docker compose logs -f llama-server   # up to 5 min for large model
docker compose logs -f cloudflared     # should show "connected"
```

### MCP Search Server

```bash
# Run tests:
cd mcp-search-server && pytest

# Run a linter check:
cd mcp-search-server && ruff check src/
```

## Architecture

```
Cloudflare Edge (api.jessfan.com)
    └─ Cloudflare Tunnel (outbound)
       └─ llama-server :8080  (llama-net internal)
          └─ cloudflared (also on llama-net)
          └─ mcp-search-server :3100 (also on llama-net)
```

| Service | Image target | GPU | Purpose |
|---|---|---|---|
| `llama-server` | `runtime` | Yes | Inference server |
| `cloudflared` | upstream image | No | Public tunnel |
| `llama-convert` | `convert` | No | Model download/quantize |
| `mcp-search-server` | custom | No | Web search MCP tool |

- **llama-server uses --webui-mcp-proxy** — it proxies MCP tool calls to mcp-search-server at port 3100. The MCP Search Server provides semantic web search with browser automation.

## Gotchas

- **`docker compose restart` won't re-read config.** After changing `.env` or `docker-compose.override.yml`, use `docker compose up -d` instead.
- **Local dev requires `docker-compose.override.yml`** to expose port 8080 and add the `host-bridge` network. This file is gitignored.
- **Model must be prepared before starting the server.** The server image does not download models — it expects a `.gguf` file in the shared `./models` volume.
- **Safetensors repos need `convert-st`, not `download`.** If the repo has no pre-built GGUF on HuggingFace, `download` will fail.
- **Windows WSL2 BSOD prevention:** If quantization or large downloads cause `CLOCK_WATCHDOG_TIMEOUT` BSODs, set `CONVERT_THREADS=4` and `CONVERT_DOWNLOAD_RATE=300M` in `.env`. For persistent issues, cap WSL2 memory globally via `%USERPROFILE%\.wslconfig`.
- **`llama-convert` needs the CUDA stub.** The Dockerfile copies `libcuda.so` from the builder stage so the binary can start without GPU access (quantization is CPU-only).
- **The `convert` stage inherits from `runtime`, not `builder`.** This means it has the CUDA runtime libs but not the build tools — the stub trick is required.
- **MCP Search Server uses a named volume** for `.cache/python` only — the full `.cache` dir (which contains Playwright browsers) must not be overwritten by an empty volume mount.
- **`mcp-search-server` healthcheck** hits `/health` on port 3100, not the SSE endpoint.
- **`LLAMA_GPU_LAYERS=99`** — llama.cpp clamps to the model's actual layer count; this is the correct way to "offload all layers".
- **`LLAMA_KV_UNIFIED=on`** — shares the full context pool across all slots (`--kv-unified`). With this default (llama.cpp >= b4550), `LLAMA_PARALLEL` controls concurrency, not context division.
- **`LLAMA_CACHE_TYPE_K=V=turbo3`** requires `LLAMA_FLASH_ATTN=on` — setting turbo3 without flash attention will fail.
- **`LLAMA_CLEAR_IDLE=on`** requires `LLAMA_CACHE_RAM` to be set and non-zero — otherwise the flag is silently omitted by `entrypoint.sh`.
- **`LLAMA_DIRECT_IO=on`** is recommended when `LLAMA_GPU_LAYERS=99` — it prevents the ~18 GB model from being cached in the OS page cache after it's already in VRAM.
- **MTP requires an MTP-capable GGUF** — the model must have `nextn`/MTP head layers baked in. Models need either `mtp_capable: True` (native MTP weights) or `mtp_graft_from: "Repo/Name"` (graft MTP from base model) in `manage_models.py`. The `--mtp` flag is blocked with a hard error for models with neither.
- **Both Qwopus fine-tunes need MTP grafting.** Neither `qwopus3.6-27b` nor `qwopus3.6-35b` ship MTP tensors in their safetensors — Unsloth's fine-tuning stripped them while leaving `mtp_num_hidden_layers: 1` in config.json. The `mtp_graft_from` field in MODELS causes `--mtp` to auto-download MTP tensors from the base model (`Qwen/Qwen3.6-27B` for the 27B dense, `Qwen/Qwen3.6-35B-A3B` for the 35B MoE) and inject them before conversion. The MTP head is architecturally independent of the fine-tuned trunk.
- **MTP + vision coexist** — MTP speculative decoding pauses during image/audio processing and resumes for text tokens. `handle_mtp_for_ubatch` detects embedding-only batches (`tokens==nullptr`) and resets its pending state so the MTP KV cache skips image positions cleanly.
- **llama-cpp source** is now `JEF1056/llama-cpp-turboquant` (`llama-next` branch) — TurboQuant KV + upstream sync + MTP speculative decoding + HIP/FATTN fixes on top.

## Environment variables (key ones)

| Variable | Default | Effect |
|---|---|---|
| `MODEL_NAME` | `qwopus3.6-27b` | Model to download via `manage_models.py` |
| `QUANT` | `Q3_K_M` | Quantization format (or `TQ_QUANT` for TurboQuant) |
| `LLAMA_MODEL` | — | Direct path to a `.gguf` file (skips auto-download) |
| `HF_TOKEN` | — | Required for gated HuggingFace repos |
| `CONVERT_THREADS` | `cpu_count//2` | CPU threads for quantization (lower to prevent BSOD) |
| `CONVERT_DOWNLOAD_RATE` | — | Throttle download speed (e.g. `300M`) to prevent BSOD |
| `LLAMA_API_KEY` | — | API key for llama-server requests |
| `CF_TUNNEL_TOKEN` | — | Cloudflare Tunnel token |
| `CF_ACCESS_HOSTNAME` | — | Cloudflare Access protected hostname |
| `LLAMA_SLOT_SAVE_PATH` | `/models/slots` | KV cache slot persistence |
| `LLAMA_KV_UNIFIED` | `on` | Share full context pool across all slots (`--kv-unified`) |
| `LLAMA_CACHE_RAM` | — | Host-RAM prompt cache in MiB (`--cache-ram`); `-1` = unlimited, `0` = off |
| `LLAMA_CLEAR_IDLE` | `on` | Save idle slots to `--cache-ram` on each new task (`--clear-idle`); requires `LLAMA_CACHE_RAM` |
| `LLAMA_SPEC_TYPE` | — | Speculative decoding type: `mtp` for ~2x speed; requires an MTP-capable GGUF |
| `LLAMA_SPEC_DRAFT_N_MAX` | `3` | Draft tokens per MTP step (3 → 86.7% acceptance on RTX 3090 at 164K ctx) |

## opencode.json

The `opencode.json` at the repo root configures the LLM provider for OpenCode sessions — it points at `http://localhost:8080/v1` with the `qwenopus3.6-27b` model and a 150K context window. No additional instructions are needed beyond what's in this file.

## Anti-Loop Discipline

These rules prevent the agent from getting stuck re-reading files or repeating tool calls.

- **Never read the same file twice in a single conversation.** If you already read a file, use what you have in context. If you need a specific section, use `grep` to find the relevant lines — do not re-read the whole file.
- **Check context before calling Read.** Before invoking any file-read tool, check whether you already have the file contents in context.
- **If you notice you're re-reading a file, stop.** Summarize what you already know, then proceed with the task instead of gathering more context.
- **Prefer targeted searches over full reads.** Use `grep`/`glob` to locate specific code before reading entire files.
- **Track which files you've read.** Keep a mental note of files already examined. If compaction prunes earlier turns, rely on summaries you've already written.

## Git operations
Prefer to use `gh` (github CLI) and `git` commands for git operations where possible, instead of searching the web.

## MCP Vs Builtin tools
Prefer MCP tools over builtin tools, like webfetch and websearch, or using shell, when available.

## 12 Rules

These rules apply to every task unless explicitly overridden. Bias: caution over speed on non-trivial work.

### Rule 1 - Think Before Coding
State assumptions explicitly. Ask rather than guess.
Push back when a simpler approach exists. Stop when confused.

### Rule 2 - Simplicity First
Minimum code that solves the problem. Nothing speculative.
No abstractions for single-use code.

### Rule 3 - Surgical Changes
Touch only what you must. Don't improve adjacent code.
Match existing style. Don't refactor what isn't broken.

### Rule 4 - Goal-Driven Execution
Define success criteria. Loop until verified.
Strong success criteria let Claude loop independently.

### Rule 5 - Use the model only for judgment calls
Use for: classification, drafting, summarization, extraction.
Do NOT use for: routing, retries, deterministic transforms.
If code can answer, code answers.

### Rule 6 - Token budgets are not advisory
Per-task: 4,000 tokens. Per-session: 30,000 tokens.
If approaching budget, summarize and start fresh.
Surface the breach. Do not silently overrun.

### Rule 7 - Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.

### Rule 8 - Read before you write
Before adding code, read exports, immediate callers, shared utilities.
If unsure why existing code is structured a certain way, ask.

### Rule 9 - Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

### Rule 10 - Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.

### Rule 11 - Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you think a convention is harmful, surface it. Don't fork silently.

### Rule 12 - Fail loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.
