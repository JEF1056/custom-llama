/**
 * Olla sticky-session affinity plugin.
 *
 * Injects a per-session `X-Olla-Session-ID` header on every LLM request so
 * Olla's sticky-session router (key_sources: ["session_header"]) pins each
 * opencode session to its own backend for KV-cache reuse across turns.
 *
 * Why the `chat.headers` hook (and not a global fetch wrapper): this hook
 * fires per request and receives the *actual* sessionID for that request.
 * Parallel subagents each run as their own child session with a distinct
 * sessionID, so each subagent gets a unique header value with no shared
 * mutable state — they fan out to different backends. A global fetch wrapper
 * with a single "current session" variable (the common pattern) would race
 * and collapse concurrent subagents onto one value.
 *
 * Requires an opencode version that exposes the `chat.headers` hook.
 */
export const OllaSession = async () => {
  return {
    "chat.headers": async (input, output) => {
      // Don't clobber an explicitly configured header.
      if (!output.headers["X-Olla-Session-ID"]) {
        output.headers["X-Olla-Session-ID"] = input.sessionID
      }
    },
  }
}

export default OllaSession
