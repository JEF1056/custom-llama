<!-- tokens: overview=950, dir=1800, modules=7200, interfaces=462, config=231 -->

---

## Project Overview

- **Language**: Python, Shell (Bash), TypeScript, YAML
- **Framework**: ik_llama.cpp (CUDA), MLX (Apple Silicon), LiteLLM Olla Router, FastMCP, OpenCode AI SDK
- **Build System**: Docker / Docker Compose (CUDA), shell scripts (Mac install), npm (OpenCode + harness plugin)
- **Entry Points**: `docker/entrypoint.sh` → `llama-server`, `mcp-search-server/src/server.py:main()`, `router/` (Olla LiteLLM proxy)
- **Architecture**: Multi-backend LLM load-balancing system — CUDA server (NVIDIA GPU) + MLX server (MacBook Apple Silicon) → Olla LiteLLM proxy → OpenCode AI SDK clients. Orchestrated by harness plugin (`github:JEF1056/harness`) providing Swarm mode with Sentinel/Orchestrator/Coder/Explorer/Reviewer/Challenger/Auditor polyglot subagent loop.

---

## Directory Structure

 ```
custom-llama/
├── `'.opencode/               # OpenCode AI SDK config + harness plugin`
│   ├── `node_modules/@opencode-ai/  # Core SDK (146 TS files: sdk, plugin, codegen)`
│   ├── `plugin/sticky-header.js   # X-Olla-Session-ID header plugin`
│   └── `plugins/olla-session.js   # Per-session sticky session pluginl`
├── `harness/                   # Harness: multi-agent swarm plugin (INLINE at /home/jfan/harness)`
│   ├── `index.ts               # Plugin entry: configures 10 agents (Sentinel, Orchestrator, Coder, Explorer, Reviewer, Challenger, Auditor, VictoryAuditor, Debugger, Researcher, Cleanup)`
│   ├── `map.ts                 # CODEBASE_MAP generator — build_codebase_map(), heuristic detection (language, framework, entry points, modules)`
│   ├── `plan.ts   `# Plan tool prompt — generic structure, does NOT reference CODEBASE_MAP.md`
│   └── `debug.ts               # Diagnostic log fetching + repair prompt`
├── `docker/                    # CUDA server: Docker files, entrypoint, compose`
│   ├── `Dockerfile             # GPU image: clones ik_llama.cpp fork | CUDA build`
│   ├── `Dockerfile.cpu         # CPU-only variant (no GPU)`
│   ├── `docker-compose.yml     # 4 services: server, server-cpu, model-prep, model-prep-cpu`
│   ├── `entrypoint.sh          # llama-server launcher (280 lines, ~30 env vars)`
│   ├── `.env / .env.example    # Server configuration (CTX, KV_TYPE, MTP, vision, ngram...)`
│   ├── `bench_longctx.py       # Long-context benchmark script`
│   └── `bench_prompt_types.py  # Prompt-processing type benchmark`
├── `mac/                       # MLX Mac deployment with LaunchAgent + supervisor
│   ├── `deploy.sh              # Fleet deploy to MacBooks (ml-2/3/4) via SSH`
│   ├── `install.sh             # Single-host install: compiles MLX, creates LaunchAgent`
│   ├── `run-mlx-server.sh      # MLX server runner`
│   ├── `launch-mlx-server.sh   # LaunchAgent wrapper`
│   ├── `supervisor-mlx-server.sh  # Health supervisor (30s interval, 10 restarts/300s)`
│   ├── `healthcheck.sh         # Server health checker`
│   ├── `bench-mlx-stress.sh    # Stress testing`
│   ├── `uninstall.sh           # Cleanup script`
│   └── `com.custom-llama.qwen36-mlx.plist.template
├── `router/                    # Olla LiteLLM proxy
│   ├── `docker-compose.yml     # Single container (ollsma) on port 4000→40114`
│   ├── `config.yaml            # Olla LB config: least-connections, sticky sessions, health checks`
│   ├── `Dockerfile             # Olla LiteLLM image`
│   └── `.env / .env.example    # Auth keys, backend URLs`
├── `scripts/                   # Offline weight pipeline
│   ├── `download-bringup.sh    # Quick bring-up: pre-quantized GGUF + mmproj (~17 GB)`
│   ├── `download-source-gguf.sh # Source BF16 GGUF download (~75 GB)`
│   ├── `prepare-weights.sh     # Phase 2 pipeline: download → imatrix → quantize`
│   ├── `quantize.sh            # Production quantization (262K-Balanced recipe)`
│   ├── `quantize-mlx.sh        # MLX-specific quantization`
│   └── `quantize-mmproj.py     # Vision projector quantization`
├── `mcp-search-server/         # Web search + browser automation MCP server`
│   ├── `Dockerfile             # MCP server container`
│   ├── `pyproject.toml         # Python: fastmcp, mcp, patchright, bs4, duckduckgo-search, ...`
│   ├── `src/`
│   │   ├── `server.py          # FastMCP server: 7 tool handlers + resources + file management`
│   │   ├── `config.py          # Settings (env vars)`
│   │   ├── `browser/           # Browser automation (patchright/Playwright)`
│   │   │   ├── `__init__.py`
│   │   │   └── `automation.py  # Click, fill, screenshot, snapshot, navigate`
│   │   ├── `search/            # Web search handlers`
│   │   │   ├── `search.py      # Fast search (titles+snippets)`
│   │   │   ├── `fetch.py       # Full page fetch (JS-rendered)`
│   │   │   └── `deep_search.py # Search + extract top 3 results`
│   │   ├── `tools/             # Additional tools`
│   │   │   ├── `code_run.py    # Sandboxed Python execution`
│   │   │   ├── `time_now.py    # Timezone-aware datetime`
│   │   │   └── `read_output.py # Pagination of large outputs`
│   │   ├── `extractor/         # Content extraction utilities`
│   │   ├── `output/            # Output formatting`
│   │   └── `output_store.py    # Persistent output storage`
│   └── `tests/                 # Test suite
├── `mlx-vlm/                   # MLX VLM model library fork (JEF1056)
│   ├── `mlx_vlm/`
│   │   ├── `models/            # 100+ model implementations (see below)`
│   │   │   ├── `llama/         # LLaMA base model`
│   │   │   ├── `llama4/        # LLaMA 4 vision`
│   │   │   ├── `qwen_3_5/       # Qwen 3.5 (with gated DeltaNet)`
│   │   │   ├── `qwen3_omni_moe/   # Audio + vision + text`
│   │   │   ├── `smollm3/       # SmollM 3`
│   │   │   ├── `granitemoe/    # Granite MoE`
│   │   │   ├── `helium/        # Helium model`
│   │   │   ├── `molmo2/        # MoMo 2`
│   │   │   ├── `kimi_k25/      # Kimi K25`
│   │   │   └── `... (80+ more model modules)`
│   │   ├── `trainer/           # LoRA, SFT, ORPO trainers`
│   │   ├── `server/            # HTTP server (Anthropic + OpenAI compatible)`
│   │   ├── `speculative/       # Speculative decoding (eagle3, dflash, MTP)`
│   │   ├── `quant/             # Quantization utilities`
│   │   ├── `tool_parsers/      # Tool calling parsers for 12+ models`
│   │   └── `*_utils.py, *_cache.py, turboquant.py`
│   └── `pyproject.toml, requirements.txt`
├── `opencode.json               # OpenCode config: 3 providers (router, llama-remote, llama-local)`
├── `docs/                      # Benchmark results & migration plan
│   ├── `qwen36-bench-results.md # RTX 3090 benchmark results
│   └── `iqllama-migration-plan.md
```

---

## Module Deep-Dives

### docker/ — CUDA Server

**Purpose**: Runs patched `ik_llama.cpp` (`JEF1056/ik_llama.cpp` fork, branch `ngram-mtp-vision-chain`) as an OpenAI-compatible API server on NVIDIA GPU.

**Key Files**:
- `Dockerfile` (GPU) / `Dockerfile.cpu` — Clones ik_llama.cpp fork, bundles CUDA toolchain
- `docker-compose.yml` (177 lines) — 4 services:
  - `server` (default): GPU server, port 8080, health check `http://localhost:8080/health`
  - `server-cpu` (profile: cpu): CPU-only, NGL=0, no MTP, no Hadamard KV
  - `model-prep` (profile: prep): One-shot pipeline — download BF16 GGUF → imatrix → quantize (needs GPU)
  - `model-prep-cpu` (profile: prep-cpu): Same pipeline on CPU
- `entrypoint.sh` (280 lines): Configures `llama-server` flags from ~30 environment variables

**Key Features** (env vars → flags):
| Default | Env Var | Description |
|---------|---------|-------------|
| 262144 | CTX | Context window in tokens (0 = auto-fit) |
| q4_0 | KV_TYPE | KV cache quantization (10 attention layers) |
| 1 | KV_HADAMARD | Hadamard-rotated K/V cache |
| 8192 | CACHE_RAM_MIB | Prompt cache RAM budget (MiB) |
| 1 | N_PARALLEL | Concurrent request slots |
| 1024 | UBATCH_SIZE | Physical batch size |
| 1 | ENABLE_VISION | Vision tower |
| 1 | ENABLE_MTP | MTP self-speculative decoding |
| 4 | MTP_N_MAX | Speculative tokens per round |
| 1 | ENABLE_NGRAM | n-gram lookup drafter |
| 0.6 | TEMP | Default temperature |

**Quantization Recipe (262K-Balanced)**:
- Edge experts (30 DeltaNet): `iq4_ks` | Middle experts: `iq3_k` | Shared expert: `q8_0`
- Attention layers: `iq5_ks` | Router: `q8_0` | Token embedding: `iq4_ks`
- Output: `q6_K` | MTP block: BF16 (output head at `q8_0`)

---

### mac/ — MLX Mac Deployment (LaunchAgent + Supervisor)

**Purpose**: Deploy MLX-based LLM server on MacBooks with Apple Silicon via LaunchAgent + supervisor.

**Key Files**:
- `install.sh` — Single-host install (any Mac with Apple Silicon): downloads MLX model, creates LaunchAgent at `~/Library/LaunchAgents/com.custom-llama.qwen36-mlx.plist`
- `deploy.sh` (125 lines) — Fleet deployment to multiple MacBooks (ml-2, ml-3, ml-4) via SSH
- `supervisor-mlx-server.sh` — Health supervisor: checks every 30s, auto-restarts up to 10 times per 300s
- `healthcheck.sh` — Quick server health check
- `com.custom-llama.qwen36-mlx.plist.template` — LaunchAgent plist template

**Deployment Modes**:
1. **Single-host**: `curl -fsSL URL | HF_TOKEN=hf_xxx bash`
2. **Fleet**: `bash mac/deploy.sh ml-2 ml-3 ml-4` (hosts: ml-2, ml-3, ml-4, configurable via `ALL_HOSTS`)

**Server Config**: Port 8080, context 229376, KV quantization bits (default: 4)

---

### router/ — Olla LiteLLM Proxy

**Purpose**: Load-balancing proxy with LLM model routing, health checks, and sticky sessions.

**Key Files**:
- `config.yaml` (77 lines): Olla LB config with:
  - **Load Balancer**: `least-connections` with sticky sessions (1h TTL, 10K max sessions)
  - **Session Key**: `X-Olla-Session-ID` header (set by `.opencode/plugins/olla-session.js`)
  - **Static Endpoints**: 4 backends (ml-1-wsl-cuda, ml-2-mlx, ml-3-mlx, ml-4-mlx), health check every 5s/2s
  - **Model Registry**: In-memory with unifier engine (24h stale threshold)
  - **Model Aliases**: `qwen3.6-35b` maps to CUDA `/models/qwen3.6-35b` + MLX `/Users/jfan/.qwen/models/qwen36-mlx`
  - **Translators**: Anthropic passthrough enabled
  - **Logging**: JSON format, info level

---

### mcp-search-server/ — Web Search + Browser Automation

**Purpose**: MCP (Model Context Protocol) server providing 7 tools for web interaction.

**Key File — `src/server.py` (520 lines)**:
```
Main entry: main() → create_server() → register_tools() → run_server()
```

**Tools Registered**:
1. `search(query)` — Fast web search (titles + snippets via DuckDuckGo)
2. `fetch(url)` — JS-rendered page text extraction (headless browser)
3. `deep_search(query)` — Search + extract full content from top 3 results
4. `browser_screenshot(url)` — Visual page capture (PNG)
5. `click(selector)` — Click element on page
6. `fill(selector, value)` — Fill input field
7. `evaluate(script)` — Run JavaScript on page
8. Additional: `navigate_page`, `get_interactables`, `take_snapshot`, `get_text`, `get_content`, `page_state`, `read_output`

**Browser Automation (`src/browser/automation.py`)**:
- Uses `patchright` (undetectable Playwright drop-in)
- Tools: navigate, screenshot, snapshot, click, fill, evaluate, interactables

**Data Flow**:
```
search/fetch → content_store → read_output (pagination)
browser → file_store (/mcp-files/) → serve_file (static files)
```

**Endpoints**:
| Method | Path | Purpose |
|--------|------|---------|
| GET | /sse | SSE stream (legacy transport) |
| POST | /mcp | Streamable HTTP transport |
| POST | / | Streamable HTTP transport |
| GET | /health | Health check |
| GET | /files | File UI |
| GET | /api/files | File list (JSON) |
| POST | /api/files/upload | File upload |
| DEL | /api/files/{filename} | File deletion |
| GET | /files/{filename} | File download |

**Auth**: Optional API key (per-IP rate limit: 5 failures / 60s → 429)

---

### mlx-vlm/ — MLX VLM Model Library

**Purpose**: JEF1056 fork of mlx-vlm supporting 100+ vision-language models.

**Key Directories**:
- `mlx_vlm/models/` — 100+ model implementations (each with config.py, model.py, [vision.py], [language.py], [processing_*.py]):
  - LLaMA family: llama/, llama4/, llama4_text/ (text-only)
  - Qwen family: qwen3_5/, qwen3_5_moe/, qwen3_omni_moe/, qwen3_vl/, qwen2_vl/
  - LLaVA family: llava/, llava_next/, llava_bunny/
  - Tiny/edge models: smollm3, moondream2, moondream3, moai_point
  - Specialized: rfdetr, rt_detr_v2, sam3, sam3_1, sam3d_body (object detection/segmentation)
  - Diffusion: ideogram4, mage_flow, flux2, nemotron_labs_diffusion
  - Others: granite_vision, helium, kimi_k25/3/kl, minicpmo, phi3/phi4, ministral, plamo, step3p7, telechat3, zaya1_vl, etc.

- `mlx_vlm/speculative/` — Speculative decoding: eagle3, dflash, MTP, drafters (deepseek_v4_mtp, gemma4_assistant, qwen3_5_mtp)
- `mlx_vlm/trainer/` — LoRA, SFT, ORPO trainers with adapter utilities
- `mlx_vlm/server/` — HTTP server: OpenAI + Anthropic compatible APIs
- `mlx_vlm/tool_parsers/` — Tool calling parsers for 12+ models
- `mlx_vlm/quant/` — Quantization utilities (AWQ, calibration)

**Tests**: 30+ test files covering models, processors, sampling, quantization, JSON tools.

---

### scripts/ — Weight Pipeline

**Purpose**: Offline weight preparation pipeline for production deployment.

**Pipeline Stages**:
1. `download-source-gguf.sh` — Download Unsloth pre-converted BF16 GGUF shards (~75 GB)
2. `prepare-weights.sh` — Run Phase 2: download → imatrix → quantize
3. `quantize.sh` — Production quantization (262K-Balanced recipe)
4. `quantize-mlx.sh` — MLX-specific quantization
5. `quantize-mmproj.py` — Vision projector (mmproj) quantization
6. `download-bringup.sh` — Quick bring-up: pre-quantized GGUF + mmproj (~17 GB)

**Pipeline**: Unsloth BF16 GGUF → compute custom imatrix → 262K-Balanced quantization → mmproj → GPU deployment

---

### opencode.json

**Purpose**: OpenCode AI SDK configuration defining LLM providers, models, and MCP integration.

**Providers**:
1. **router** (`router-master-key`) → `http://coolify:4000/olla/proxy/v1` → `qwen3.6-35b` (Qwen3.6-35B-A3B via Olla proxy)
2. **llama-remote** (`sk-noauth`) → `http://ml-2:8081/v1` → Qwen3.6-35B-A3B 262K-Remote
3. **llama-local** → `http://localhost:8080/v1` → Qwen3.6-35B-A3B 262K-Local

**Model Config**: 229,376 context limit, 8,192 output limit, tools + attachments enabled

**Features**:
- Plugin: `github:JEF1056/harness` + sticky-header plugin (`X-Olla-Session-ID`)
- MCP: Remote SSE connection to `mcp-search-server` at `http://localhost:3100/sse`

---

### harness/ — Swarm Plugin (source at /home/jfan/harness)

**Purpose**: Multi-agent swarm orchestrator plugin for OpenCode. Provides 4 slash commands (`/harness`, `/plan`, `/map`, `/debug`) and configures 10 agent subtypes with distinct prompts and execution semantics.

**4 Slash Commands**:
- `/harness` — Launches Swarm mode: Sentinel runs on main thread, spawns subagents (Explorer, Coder, Reviewer, Challenger, Auditor, VictoryAuditor, Debugger, Researcher, Cleanup) via `task` tool, polls with `task_status`
- `/map` — Generates `CODEBASE_MAP.md` via `map.ts`'s `build_codebase_map()`: heuristic language detection, framework scoring, entry point discovery, module identification, directory tree building. Then spawns an Explorer agent to validate/enrich the map
- `/plan` — Writes a plan to `.agents/plans/<name>.md` with generic structure (overview, design, steps, files, verification). **Does NOT reference `CODEBASE_MAP.md`**
- `/debug` — Fetches diagnostic logs and runs repair prompt for a target_id

**Key Files**:
- `index.ts` (1003 lines) — Plugin entry point:
  - `config()` phase: Registers 10 subagent types with prompts + model resolution
  - `command.execute.before` handler: Processes /harness, /plan, /debug, /map commands
  - Workspace locking via `.agents/lock.json` (race-free exclusive creation, TTL-based)
  - Heartbeat monitoring: Watches `progress.md` for stale agents (>5 min), broadcasts toasts
  - File watchers: Monitors agent folders for `progress.md`, `handoff.md`, `escalation.md` changes
- `map.ts` (662 lines) — Codebase map builder:
  - `build_codebase_map()`: Builds directory tree, identifies key files, entry points, modules
  - Heuristic detection: language scoring (weighted by depth), framework detection (package.json deps + Docker), build system detection
  - Module detection: Top-level `package.json`/`pyproject.toml`/`Cargo.toml`/`go.mod` or named feature dirs (`src`, `models/`, etc.)
  - Known weakness: `detectFramework()` scores `Cargo.toml` as "Rust" even when zero `.rs` files exist (false positive)
- `plan.ts` (32 lines) — Plan tool prompt template
- `debug.ts` — Diagnostic log fetching + repair prompt

---

### @opencode-ai/sdk (local node_modules)

**Purpose**: Core OpenCode SDK providing the plugin API (`Plugin`, `PluginInput`, `PluginOptions`, `tool`), auto-generated types, and OpenAI-compatible provider for routing requests.

**Structure**:
- `@opencode-ai/plugin`: Plugin interface, `tool`, `config` hooks
- `@opencode-ai/sdk`: Auto-generated API types/v2 endpoints, client model
- `@ai-sdk/openai-compatible`: OpenAI-compatible provider

---

## Codebase Statistics

| Module | Files (excl. node_modules) | Lines | Language |
|--------|---------------------------|-------|----------|
| docker/ | 8 | ~1,126 | Shell, Docker, Python |
| mac/ | 10 | ~500 | Shell |
| router/ | 4 | ~100 | YAML, Docker |
| scripts/ | 6 | ~400 | Shell, Python |
| mcp-search-server/ | 40+ | ~2,000 | Python |
| mlx-vlm/ | 400+ | ~15,000 | Python |
| harness/ | 5 | ~2,200 | TypeScript |
| @opencode-ai/ | ~146 | ~5,000 | TypeScript |

---

## Architecture Diagram

 ```
┌──────────────────────────────────────────────────────────────┐
│  OpenCode AI SDK Client                                      │
│  (provider: router → qwen3.6-35b)                           │
└──────────────────────────┬───────────────────────────────────┘
                           │ OpenAI-compatible API
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Olla LiteLLM Proxy (router/, port 4000)                    │
│  Load Balancer: least-connections, sticky sessions          │
│  – ml-1-wsl-cuda (CUDA, priority 100)                       │
│  – ml-2-mlx (MLX Mac, priority 95)                          │
│  – ml-3-mlx (MLX Mac, priority 90)                          │
│  – ml-4-mlx (MLX Mac, priority 50)                          │
└──────┬────────────┬────────────┬─────────────────────────────┘
       │            │            │
       ▼            ▼            ▼
┌──────────┐ ┌────────────┐ ┌────────────┐
│ CUDA     │ │ MLX Mac    │ │ MLX Mac    │
│ Server   │ │ Server     │ │ Server     │
│ :8080    │ │ :8081      │ │ :8082      │
│ ik_llama │ │ MLX VLM    │ │ MLX VLM    │
│ patched  │ │ (mlx-vlm)  │ │ (mlx-vlm)  │
└──────────┘ └────────────┘ └────────────┘

┌──────────────────────────────────────────────────────────────┐
│  MCP Search Server (mcp-search-server/, port 3100)          │
│  7 tools: search, fetch, deep_search, browser,              │
│           code_run, time_now, read_output                   │
│  Browser: patchright (undetectable Playwright)              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Harness Plugin (INLINE: /home/jfan/harness)                │
│  Commands: /harness, /plan({plan}), /map({map}), /debug({debug})│
│  Agents: Sentinel, Orchestrator, Coder, Explorer, Reviewer, │
│          Challenger, Auditor, VictoryAuditor, Debugger,     │
│          Researcher, Cleanup                                │
└──────────────────────────────────────────────────────────────┘
 ```

---

## Recent Changes

- Last regenerated: 2026-07-31T00:00:00.000Z
- Scope: Full project
