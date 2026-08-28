# OpenCode + Olla Sticky Session Setup

This folder contains the installer, configuration templates, and sticky-session affinity plugins for [OpenCode](https://opencode.ai) to route to the unified Olla router cluster with persistent KV-cache reuse across turns and parallel subagents.

---

## Quick Start

### Install into Current Repository
```bash
./install.sh
```

### Install into Any Other Repository
```bash
./install.sh /path/to/any-repo
```

### Install Globally (User Configuration)
```bash
./install.sh --global
```

---

## What Gets Installed

1. **`opencode.json`**
   - Configures the `olla` provider pointing to `http://coolify:4000/olla/openai/v1`
   - Sets primary model to `olla/qwen3.8-27b`
   - Configures fallback endpoints for `llama-ml1`, `llama-ml2`, and `llama-ml3`
   - Enables MCP search server (`http://localhost:3100/`)

2. **`.opencode/plugins/olla-session.js`**
   - Hooks into `chat.headers` per request to inject `X-Olla-Session-ID: <sessionID>`.
   - Olla's sticky-session router pins each conversation and child subagent to its dedicated backend node to maximize prompt cache hits.
