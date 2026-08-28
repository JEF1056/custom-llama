/**
 * Generic sticky-session header plugin.
 *
 * Injects a configurable header per LLM request, pinning each session to its
 * own backend for KV-cache reuse (or any other sticky-session proxy that reads
 * a session ID from a custom header).
 *
 * Register from opencode.json:
 *   "plugin": ["./.opencode/plugin/sticky-header.js", { "headerName": "X-Olla-Session-ID" }]
 *
 * Falls back to `X-Olla-Session-ID` if no options are given.
 *
 * Why `chat.headers`: fires per request with the actual sessionID.
 * Parallel subagents get distinct values, avoiding race conditions on shared variables.
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
