/**
 * SSE streaming parser for OpenAI-compatible chat completions.
 * Handles <think>…</think> extraction and tool_calls delta accumulation.
 */
import type { StreamResult, ToolCall, ToolCallDraft } from '../types'

// ── <think> state machine ────────────────────────────────────────────────────

interface ThinkState {
  inThinking: boolean
  thinkBuffer: string
  contentBuffer: string
  // partial tag match buffer (handles chunk boundaries like "<thi" + "nk>")
  tagBuffer: string
}

function makeThinkState(): ThinkState {
  return { inThinking: false, thinkBuffer: '', contentBuffer: '', tagBuffer: '' }
}

const OPEN_TAG = '<think>'
const CLOSE_TAG = '</think>'

/**
 * Feed a raw text chunk through the <think> state machine.
 * Returns { content, thinking } increments for this chunk.
 */
function processChunk(state: ThinkState, chunk: string): { content: string; thinking: string } {
  let content = ''
  let thinking = ''
  let i = 0

  while (i < chunk.length) {
    const ch = chunk[i]

    if (!state.inThinking) {
      // Try to match OPEN_TAG
      if (OPEN_TAG.startsWith(state.tagBuffer + ch)) {
        state.tagBuffer += ch
        if (state.tagBuffer === OPEN_TAG) {
          state.inThinking = true
          state.tagBuffer = ''
        }
        i++
        continue
      } else if (state.tagBuffer.length > 0) {
        // Flush buffered non-tag chars
        content += state.tagBuffer
        state.tagBuffer = ''
        // Re-process current char without advancing
        continue
      } else {
        content += ch
        i++
      }
    } else {
      // Try to match CLOSE_TAG
      if (CLOSE_TAG.startsWith(state.tagBuffer + ch)) {
        state.tagBuffer += ch
        if (state.tagBuffer === CLOSE_TAG) {
          state.inThinking = false
          state.tagBuffer = ''
        }
        i++
        continue
      } else if (state.tagBuffer.length > 0) {
        // Flush tag buffer to thinking
        thinking += state.tagBuffer
        state.tagBuffer = ''
        continue
      } else {
        thinking += ch
        i++
      }
    }
  }

  return { content, thinking }
}

// ── Tool call draft accumulator ──────────────────────────────────────────────

function mergeDraft(drafts: Map<number, ToolCallDraft>, delta: NonNullable<DeltaToolCall>): void {
  const idx = delta.index
  if (!drafts.has(idx)) {
    drafts.set(idx, { id: delta.id ?? '', name: delta.function?.name ?? '', arguments: '' })
  }
  const d = drafts.get(idx)!
  if (delta.id) d.id = delta.id
  if (delta.function?.name) d.name += delta.function.name
  if (delta.function?.arguments) d.arguments += delta.function.arguments
}

function draftsToToolCalls(drafts: Map<number, ToolCallDraft>): ToolCall[] {
  return [...drafts.values()].map((d) => ({
    id: d.id,
    type: 'function' as const,
    function: { name: d.name, arguments: d.arguments },
  }))
}

// ── SSE line parser ──────────────────────────────────────────────────────────

interface DeltaToolCall {
  index: number
  id?: string
  function?: { name?: string; arguments?: string }
}

interface StreamDelta {
  content?: string | null
  tool_calls?: DeltaToolCall[]
}

interface SSEChoice {
  delta: StreamDelta
  finish_reason?: string | null
}

function parseLine(line: string): { delta?: StreamDelta; finishReason?: string | null } | null {
  if (!line.startsWith('data: ')) return null
  const payload = line.slice(6).trim()
  if (payload === '[DONE]') return { finishReason: 'done' }
  try {
    const json = JSON.parse(payload)
    const choice: SSEChoice | undefined = json.choices?.[0]
    if (!choice) return null
    return { delta: choice.delta, finishReason: choice.finish_reason }
  } catch {
    return null
  }
}

// ── Main stream consumer ─────────────────────────────────────────────────────

export type StreamCallbacks = {
  onContent?: (text: string) => void
  onThinking?: (text: string) => void
  onToolCallDelta?: (index: number, argChunk: string) => void
}

/**
 * Consume an OpenAI SSE stream and return the accumulated StreamResult.
 * Fires optional callbacks for real-time UI updates.
 */
export async function consumeStream(
  response: Response,
  signal: AbortSignal,
  callbacks: StreamCallbacks = {},
): Promise<StreamResult> {
  if (!response.body) throw new Error('Response has no body')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  const thinkState = makeThinkState()
  const toolDrafts = new Map<number, ToolCallDraft>()

  let rawContent = ''
  let thinkAccum = ''
  let lineBuffer = ''
  let finishReason: StreamResult['finishReason'] = null

  try {
    while (true) {
      if (signal.aborted) break

      const { done, value } = await reader.read()
      if (done) break

      lineBuffer += decoder.decode(value, { stream: true })

      const lines = lineBuffer.split('\n')
      lineBuffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trimEnd()
        if (!trimmed) continue

        const parsed = parseLine(trimmed)
        if (!parsed) continue

        if (parsed.finishReason === 'done') continue

        if (parsed.finishReason && parsed.finishReason !== 'done') {
          finishReason = parsed.finishReason as StreamResult['finishReason']
        }

        const { delta } = parsed
        if (!delta) continue

        // Tool call deltas
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            mergeDraft(toolDrafts, tc)
            if (tc.function?.arguments && callbacks.onToolCallDelta) {
              callbacks.onToolCallDelta(tc.index, tc.function.arguments)
            }
          }
        }

        // Text content (may contain <think>)
        if (delta.content) {
          const { content, thinking } = processChunk(thinkState, delta.content)

          if (content) {
            rawContent += content
            callbacks.onContent?.(content)
          }
          if (thinking) {
            thinkAccum += thinking
            callbacks.onThinking?.(thinking)
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  const toolCalls = draftsToToolCalls(toolDrafts)

  // If we have tool calls but finishReason was never set, infer it
  if (toolCalls.length > 0 && !finishReason) {
    finishReason = 'tool_calls'
  }

  return {
    content: rawContent,
    thinking: thinkAccum,
    toolCalls,
    finishReason: finishReason ?? 'stop',
  }
}

/**
 * Strip thinking content from a message before sending to API.
 * Returns message content with <think> blocks removed.
 */
export function stripThinkingFromContent(content: string): string {
  return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
}
