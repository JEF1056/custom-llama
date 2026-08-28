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
