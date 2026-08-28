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
