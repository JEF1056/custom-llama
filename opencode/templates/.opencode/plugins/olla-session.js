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
