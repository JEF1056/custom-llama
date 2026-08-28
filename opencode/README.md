# OpenCode + Olla Setup

Standalone installer, configurations, and plugins for [OpenCode](https://opencode.ai) to route to the unified Olla router cluster with persistent KV-cache reuse, token-per-second performance metrics, and harness integration.

---

## One-Line Install via `curl` (No Clone Required)

### 1. Install into Current Repository
```bash
curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash
```

### 2. Install into Any Specific Repository
```bash
curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash -s -- /path/to/repo
```

### 3. Install Globally (`~/.config/opencode`)
```bash
curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash -s -- --global
```

---

## Local Execution
If you already have this repo cloned:
```bash
./opencode/install.sh [TARGET_DIR]
```

---

## Included Plugins & Configurations

1. **`github:JEF1056/harness`**
   - Autonomous agent harness plugin.

2. **`.opencode/plugins/olla-session.js` (Persistent Session Affinity)**
   - Injects `X-Olla-Session-ID: <sessionID>` per request.
   - Pinned session routing across conversation turns and parallel subagents for continuous KV-cache hits.

3. **`.opencode/plugins/tps.js` (Tokens-Per-Second Metric)**
   - Tracks response latency, completion tokens, prompt tokens, and real-time generation speed (`⚡ [Olla TPS] ... tok/s`).

4. **`opencode.json`**
   - Configures `olla` provider pointing to `http://coolify:4000/olla/openai/v1` (`olla/qwen3.8-27b`).
   - Configures direct fallback endpoints for `llama-ml1`, `llama-ml2`, and `llama-ml3`.
   - Connects to local MCP search server (`http://localhost:3100/`).
