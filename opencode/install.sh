#!/usr/bin/env bash
#
# OpenCode + Olla Cluster Installer
# Standalone installer for opencode.json, sticky session affinity, harness, and TPS tracking.
# Can be run locally or streamed directly via curl:
#   curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash
#
set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_help() {
    cat << 'EOF'
Usage:
  ./install.sh [TARGET_DIR] [OPTIONS]
  curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash -s -- [TARGET_DIR] [OPTIONS]

Arguments:
  TARGET_DIR       Directory to install into (default: current working directory)

Options:
  -g, --global     Install globally into ~/.config/opencode
  -f, --force      Overwrite existing opencode.json without backup
  -h, --help       Show this help message

Examples:
  # Install into current repository
  curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash

  # Install into specific repository
  curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash -s -- /path/to/repo

  # Install globally
  curl -sSL https://raw.githubusercontent.com/JEF1056/custom-llama/hosting/opencode/install.sh | bash -s -- --global
EOF
}

TARGET_DIR=""
GLOBAL_MODE=0
FORCE_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--global)
            GLOBAL_MODE=1
            shift
            ;;
        -f|--force)
            FORCE_MODE=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$TARGET_DIR" ]]; then
                TARGET_DIR="$1"
            else
                echo -e "${RED}Error: Unknown argument: $1${NC}" >&2
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ $GLOBAL_MODE -eq 1 ]]; then
    TARGET_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
elif [[ -z "$TARGET_DIR" ]]; then
    TARGET_DIR="$(pwd)"
fi

# Resolve target directory
mkdir -p "$TARGET_DIR"
TARGET_DIR="$(cd "$TARGET_DIR" 2>/dev/null && pwd || echo "$TARGET_DIR")"

echo -e "${BLUE}=== OpenCode + Olla Cluster Installer ===${NC}"
echo -e "Target Directory: ${GREEN}$TARGET_DIR${NC}"

# -----------------------------------------------------------------------------
# Clean up prior installation artifacts for a fresh install
# -----------------------------------------------------------------------------
PLUGIN_DIR="$TARGET_DIR/.opencode/plugins"
LEGACY_PLUGIN_DIR="$TARGET_DIR/.opencode/plugin"

echo -e "${YELLOW}Cleaning up prior installation artifacts for fresh install...${NC}"
rm -rf "$PLUGIN_DIR" "$LEGACY_PLUGIN_DIR"
rm -f "$TARGET_DIR/opencode.json.bak"*
mkdir -p "$PLUGIN_DIR" "$LEGACY_PLUGIN_DIR"

# -----------------------------------------------------------------------------
# Embedded Plugin: olla-session.js
# -----------------------------------------------------------------------------
cat << 'EOF' > "$PLUGIN_DIR/olla-session.js"
/**
 * Olla sticky-session affinity plugin for OpenCode.
 *
 * Injects a per-session `X-Olla-Session-ID` header on every LLM request so
 * Olla's sticky-session router (key_sources: ["session_header"]) pins each
 * OpenCode session to its own backend for KV-cache reuse across turns.
 *
 * Why the `chat.headers` hook: this hook fires per request and receives the
 * actual `sessionID` for that request. Parallel subagents each run as their
 * own child session with a distinct sessionID, ensuring each subagent gets
 * a unique header value with no shared mutable state.
 *
 * Requires an OpenCode version that exposes the `chat.headers` hook.
 */
export const OllaSession = async () => {
  return {
    "chat.headers": async (input, output) => {
      if (!output.headers) {
        output.headers = {}
      }
      // Don't clobber an explicitly configured header
      if (!output.headers["X-Olla-Session-ID"]) {
        output.headers["X-Olla-Session-ID"] = input.sessionID || `session-${Date.now()}`
      }
    },
  }
}

export default OllaSession
EOF
echo -e "  ${GREEN}✓${NC} Installed fresh plugin: ${CYAN}.opencode/plugins/olla-session.js${NC}"

# -----------------------------------------------------------------------------
# Embedded Plugin: tps.js
# -----------------------------------------------------------------------------
cat << 'EOF' > "$PLUGIN_DIR/tps.js"
/**
 * TPS (Tokens Per Second) & Generation Performance Tracker for OpenCode.
 *
 * Measures prompt processing time, completion latency, total tokens,
 * and calculates output tokens/second (TPS) on every turn.
 */
export const TpsPlugin = async () => {
  const requestTimers = new Map()

  return {
    "chat.headers": async (input, output) => {
      const key = input.sessionID || "default"
      requestTimers.set(key, {
        start: performance.now(),
        date: Date.now(),
      })
    },
    "chat.response": async (input, output) => {
      const key = input.sessionID || "default"
      const timer = requestTimers.get(key)
      if (!timer) return

      const durationSec = (performance.now() - timer.start) / 1000
      requestTimers.delete(key)

      const usage = output?.usage || output?.response?.usage
      if (usage && durationSec > 0) {
        const promptTokens = usage.prompt_tokens ?? usage.promptTokens ?? 0
        const compTokens = usage.completion_tokens ?? usage.completionTokens ?? 0
        const totalTokens = usage.total_tokens ?? usage.totalTokens ?? (promptTokens + compTokens)
        const tps = compTokens > 0 ? (compTokens / durationSec).toFixed(1) : "0.0"

        console.log(
          `⚡ [TPS] ${compTokens} tokens in ${durationSec.toFixed(2)}s (${tps} tok/s) | prompt: ${promptTokens} tok | total: ${totalTokens} tok`
        )
      }
    },
  }
}

export default TpsPlugin
EOF
echo -e "  ${GREEN}✓${NC} Installed fresh plugin: ${CYAN}.opencode/plugins/tps.js${NC}"

# -----------------------------------------------------------------------------
# Embedded Plugin: sticky-header.js (legacy/configurable fallback)
# -----------------------------------------------------------------------------
cat << 'EOF' > "$LEGACY_PLUGIN_DIR/sticky-header.js"
/**
 * Generic sticky-session header plugin.
 *
 * Injects a configurable header per LLM request, pinning each session to its
 * own backend for KV-cache reuse (or any other sticky-session proxy that reads
 * a session ID from a custom header).
 */
export const StickyHeader = async (options = {}) => {
  const headerName = options.headerName || "X-Olla-Session-ID"
  return {
    "chat.headers": async (input, output) => {
      if (!output.headers) {
        output.headers = {}
      }
      if (!output.headers[headerName]) {
        output.headers[headerName] = input.sessionID || `session-${Date.now()}`
      }
    },
  }
}

export default StickyHeader
EOF
echo -e "  ${GREEN}✓${NC} Installed fresh plugin: ${CYAN}.opencode/plugin/sticky-header.js${NC}"

# -----------------------------------------------------------------------------
# Embedded Config: opencode.json
# -----------------------------------------------------------------------------
CONFIG_DEST="$TARGET_DIR/opencode.json"
if [[ -f "$CONFIG_DEST" && $FORCE_MODE -eq 0 ]]; then
    BACKUP_DEST="$CONFIG_DEST.bak.$(date +%s)"
    echo -e "${YELLOW}Backing up existing config -> $BACKUP_DEST${NC}"
    cp "$CONFIG_DEST" "$BACKUP_DEST"
fi

cat << 'EOF' > "$CONFIG_DEST"
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "github:JEF1056/harness",
    "./.opencode/plugins/olla-session.js",
    "./.opencode/plugins/tps.js"
  ],
  "provider": {
    "olla": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "olla",
      "options": {
        "baseURL": "http://coolify:4000/olla/openai/v1",
        "apiKey": "router-master-key"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B (Olla Cluster)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 131072,
            "output": 8192
          }
        },
        "qwen3.6-35b": {
          "name": "Qwen3.6-35B (Olla Cluster)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 262144,
            "output": 8192
          }
        }
      }
    },
    "ml1": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ml1",
      "options": {
        "baseURL": "http://100.118.67.28:8080/v1",
        "apiKey": "sk-noauth"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B-CUDA (ml-1-wsl Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 262144,
            "output": 8192
          }
        },
        "/models/qwen3.8-27b": {
          "name": "Qwen3.8-27B-CUDA (ml-1-wsl Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 262144,
            "output": 8192
          }
        }
      }
    },
    "ml2": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ml2",
      "options": {
        "baseURL": "http://100.77.84.65:8080/v1",
        "apiKey": "sk-noauth"
      },
      "models": {
        "trohrbaugh/Qwen3.8-27B-heretic-ara": {
          "name": "Qwen3.8-27B-MLX (ml-2 Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 131072,
            "output": 8192
          }
        },
        "/Users/jfan/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4": {
          "name": "Qwen3.8-27B-MXFP4 (ml-2 Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 131072,
            "output": 8192
          }
        }
      }
    },
    "ml3": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ml3",
      "options": {
        "baseURL": "http://100.93.207.60:8080/v1",
        "apiKey": "sk-noauth"
      },
      "models": {
        "trohrbaugh/Qwen3.8-27B-heretic-ara": {
          "name": "Qwen3.8-27B-MLX (ml-3 Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 131072,
            "output": 8192
          }
        },
        "/Users/jfan/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4": {
          "name": "Qwen3.8-27B-MXFP4 (ml-3 Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 131072,
            "output": 8192
          }
        }
      }
    },
    "llama-ml1": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-ml1",
      "options": {
        "baseURL": "http://100.118.67.28:8080/v1",
        "apiKey": "sk-noauth"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B-CUDA (ml-1-wsl Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 262144,
            "output": 8192
          }
        },
        "/models/qwen3.8-27b": {
          "name": "Qwen3.8-27B-CUDA (ml-1-wsl Direct)",
          "tools": true,
          "attachment": true,
          "limit": {
            "context": 262144,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "olla/qwen3.8-27b",
  "mcp": {
    "mcp-search-server": {
      "type": "remote",
      "enabled": true,
      "url": "http://localhost:3100/"
    }
  }
}
EOF
echo -e "  ${GREEN}✓${NC} Installed fresh config: ${CYAN}opencode.json${NC}"

echo -e "\n${GREEN}=== Fresh Installation Complete! ===${NC}"
echo -e "• Default Model: ${BLUE}olla/qwen3.8-27b${NC} (routed cluster with session affinity)"
echo -e "• Direct Models: ${BLUE}ml1/qwen3.8-27b${NC} (direct to ml-1-wsl CUDA, bypasses Olla)"
echo -e "                 ${BLUE}ml2/trohrbaugh/Qwen3.8-27B-heretic-ara${NC} (direct to ml-2 MLX)"
echo -e "                 ${BLUE}ml3/trohrbaugh/Qwen3.8-27B-heretic-ara${NC} (direct to ml-3 MLX)"
echo -e "• Plugins:       ${BLUE}github:JEF1056/harness${NC}, ${BLUE}olla-session${NC}, ${BLUE}tps${NC}"
