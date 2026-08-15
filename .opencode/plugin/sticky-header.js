/**
 * Generic sticky-session header plugin.
 *
 * Injects a configurable header per LLM request, pinning each session to its
 * own backend for KV-cache reuse (or any other sticky-session proxy that reads
 * a session ID from a custom header).
 *
 * Register from opencode.json:
 *   "plugin": ["./plugin/sticky-header", { "headerName": "X-My-Session" }]
 *
 * Falls back to `X-Session-ID` if no options are given.
 *
 * Why `chat.headers` (not a global fetch wrapper): fires per request with the
 * actual sessionID. Parallel subagents get distinct values, no racing on a
 * shared variable.
 */
export const StickyHeader = async (options = {}) => {
  const headerName = options.headerName || "X-Session-ID"
  return {
    "chat.headers": async (input, output) => {
      if (!output.headers[headerName]) {
        output.headers[headerName] = input.sessionID
      }
    },
  }
}

export default StickyHeader
