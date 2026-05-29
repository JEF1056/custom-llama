/**
 * Core chat hook: agentic loop, streaming, tool dispatch, abort.
 */
import { useState, useCallback, useRef } from 'react'
import type { Message, ContentPart, ProcessedAttachment, Settings } from '../types'
import { useSettingsStore } from '../store/settingsStore'
import { useChatStore } from '../store/chatStore'
import { consumeStream, stripThinkingFromContent } from '../lib/stream'
import { buildAttachmentInjection } from '../lib/fileProcessor'
import type { UseMCPResult } from './useMCP'

// ── Request builder ───────────────────────────────────────────────────────────

function stripThinkingFromMessages(messages: Message[]): Message[] {
  return messages.map((m) => {
    if (m.role !== 'assistant' || typeof m.content !== 'string') return m
    return { ...m, content: stripThinkingFromContent(m.content), thinking: undefined }
  })
}

function toApiMessages(messages: Message[], systemPrompt: string) {
  const out: Array<{
    role: string
    content: string | ContentPart[]
    tool_calls?: unknown
    tool_call_id?: string
  }> = [{ role: 'system', content: systemPrompt }]

  for (const m of messages) {
    if (m.role === 'tool') {
      out.push({ role: 'tool', content: typeof m.content === 'string' ? m.content : '', tool_call_id: m.tool_call_id })
    } else if (m.role === 'assistant' && m.tool_calls?.length) {
      out.push({ role: 'assistant', content: typeof m.content === 'string' ? m.content : '', tool_calls: m.tool_calls })
    } else {
      out.push({ role: m.role, content: m.content })
    }
  }

  return out
}

// ── Stream completion ─────────────────────────────────────────────────────────

async function streamCompletion(
  apiMessages: ReturnType<typeof toApiMessages>,
  settings: Settings,
  tools: unknown[],
  signal: AbortSignal,
  onContent: (c: string) => void,
  onThinking: (t: string) => void,
) {
  const body: Record<string, unknown> = {
    model: settings.model || undefined,
    messages: apiMessages,
    stream: true,
    temperature: settings.temperature,
    max_tokens: settings.maxTokens,
    top_p: settings.topP,
    top_k: settings.topK,
    frequency_penalty: settings.frequencyPenalty,
    presence_penalty: settings.presencePenalty,
  }

  if (tools.length > 0) {
    body.tools = tools
    body.tool_choice = 'auto'
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  if (settings.apiKey) headers['Authorization'] = `Bearer ${settings.apiKey}`

  const res = await fetch(`${settings.apiBaseUrl}/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API error ${res.status}: ${text}`)
  }

  return consumeStream(res, signal, { onContent, onThinking })
}

// ── useChat ───────────────────────────────────────────────────────────────────

export interface UseChatResult {
  isStreaming: boolean
  agentIteration: number | null
  sendMessage: (text: string, attachments: ProcessedAttachment[]) => Promise<void>
  abort: () => void
}

export function useChat(convId: string | null, mcp: UseMCPResult): UseChatResult {
  const { settings } = useSettingsStore()
  const { appendMessage, updateMessage, appendToolResult, getConversation } = useChatStore()
  const [isStreaming, setIsStreaming] = useState(false)
  const [agentIteration, setAgentIteration] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const abort = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const sendMessage = useCallback(
    async (text: string, attachments: ProcessedAttachment[]) => {
      if (!convId || isStreaming) return

      // Build user message content
      const injections: string[] = []
      const imageParts: ContentPart[] = []

      for (const att of attachments) {
        const injection = buildAttachmentInjection(att)
        if (injection) injections.push(injection)

        if (att.kind === 'image') {
          imageParts.push({
            type: 'image_url',
            image_url: { url: `data:${att.mimeType};base64,${att.base64}` },
          })
        } else if (att.kind === 'pdf') {
          for (const page of att.pages) {
            imageParts.push({
              type: 'image_url',
              image_url: { url: `data:image/png;base64,${page}` },
            })
          }
        }
      }

      const textWithInjections = [...injections, text].filter(Boolean).join('\n\n')

      let userContent: string | ContentPart[]
      if (imageParts.length > 0) {
        userContent = [
          ...imageParts,
          { type: 'text', text: textWithInjections },
        ]
      } else {
        userContent = textWithInjections
      }

      // Append user message to store
      appendMessage(convId, { role: 'user', content: userContent })

      // Build thinking directive
      const thinkDirective = settings.enableThinking ? ' /think' : ' /no_think'
      const systemPrompt = settings.systemPrompt + thinkDirective

      setIsStreaming(true)
      setAgentIteration(null)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const conv = getConversation(convId)
        if (!conv) return

        // Working message history for the loop (includes new user message)
        let loopMessages = stripThinkingFromMessages(conv.messages)

        let iteration = 0

        while (iteration < settings.maxAgentIterations) {
          iteration++
          setAgentIteration(iteration)

          const apiMessages = toApiMessages(loopMessages, systemPrompt)
          const tools = settings.enableMcp ? mcp.tools : []

          // Reserve a message slot for streaming assistant output
          const assistantMsgId = appendMessage(convId, {
            role: 'assistant',
            content: '',
            streaming: true,
            agentIteration: iteration,
          })

          let streamedContent = ''
          let streamedThinking = ''
          let charCount = 0
          const streamStart = Date.now()

          const result = await streamCompletion(
            apiMessages,
            settings,
            tools,
            controller.signal,
            (chunk) => {
              streamedContent += chunk
              charCount += chunk.length
              const elapsed = (Date.now() - streamStart) / 1000
              const tokenCount = Math.round(charCount / 4)
              const tokensPerSec = elapsed > 0.2 ? Math.round(tokenCount / elapsed) : 0
              updateMessage(convId, assistantMsgId, { content: streamedContent, streaming: true, tokenCount, tokensPerSec })
            },
            (chunk) => {
              streamedThinking += chunk
              updateMessage(convId, assistantMsgId, { thinking: streamedThinking, streaming: true })
            },
          )

          // Finalize: freeze the last tok/s so it stays visible after streaming
          const totalElapsed = (Date.now() - streamStart) / 1000
          const finalTokenCount = Math.round(charCount / 4)
          const finalTokPerSec = totalElapsed > 0.2 ? Math.round(finalTokenCount / totalElapsed) : 0

          // Finalize assistant message
          updateMessage(convId, assistantMsgId, {
            content: result.content || streamedContent,
            thinking: result.thinking || streamedThinking || undefined,
            tool_calls: result.toolCalls.length > 0 ? result.toolCalls : undefined,
            streaming: false,
            tokenCount: finalTokenCount,
            tokensPerSec: finalTokPerSec,
          })

          // Stop conditions
          if (result.finishReason === 'stop' || result.finishReason === 'length') break
          if (controller.signal.aborted) break

          if (result.finishReason === 'tool_calls' && result.toolCalls.length > 0) {
            // Add assistant tool-call message to loop history
            loopMessages = [
              ...loopMessages,
              {
                id: assistantMsgId,
                role: 'assistant' as const,
                content: result.content,
                tool_calls: result.toolCalls,
                timestamp: Date.now(),
              },
            ]

            // Execute each tool call
            for (const tc of result.toolCalls) {
              const toolResultContent = await resolveToolCall(tc.function.name, tc.function.arguments, mcp)
              const isError = toolResultContent.startsWith('Error:')

              // Append role=tool message to store (for history re-send)
              const toolMsgId = appendMessage(convId, {
                role: 'tool',
                content: toolResultContent,
                tool_call_id: tc.id,
              })

              // Update assistant message's toolResults for UI display
              appendToolResult(convId, tc.id, {
                toolCallId: tc.id,
                content: toolResultContent,
                isError,
              })

              loopMessages = [
                ...loopMessages,
                {
                  id: toolMsgId,
                  role: 'tool' as const,
                  content: toolResultContent,
                  tool_call_id: tc.id,
                  timestamp: Date.now(),
                },
              ]
            }

            // Continue loop for next assistant turn
            continue
          }

          // No tool calls and not stop — break to avoid infinite loop
          break
        }

        if (iteration >= settings.maxAgentIterations) {
          appendMessage(convId, {
            role: 'assistant',
            content: `⚠️ Max agentic iterations (${settings.maxAgentIterations}) reached. The task may be incomplete.`,
          })
        }
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          appendMessage(convId, {
            role: 'assistant',
            content: `❌ Error: ${err instanceof Error ? err.message : String(err)}`,
          })
        }
      } finally {
        setIsStreaming(false)
        setAgentIteration(null)
        abortRef.current = null
      }
    },
    [convId, isStreaming, settings, mcp, appendMessage, updateMessage, appendToolResult, getConversation],
  )

  return { isStreaming, agentIteration, sendMessage, abort }
}

// ── Tool call resolution ──────────────────────────────────────────────────────

async function resolveToolCall(
  name: string,
  argumentsJson: string,
  mcp: UseMCPResult,
): Promise<string> {
  if (!mcp.knownToolNames.has(name)) {
    return `Error: unknown tool "${name}". Available: ${[...mcp.knownToolNames].join(', ')}`
  }

  let args: Record<string, unknown>
  try {
    args = JSON.parse(argumentsJson)
  } catch {
    return `Error: invalid JSON arguments: ${argumentsJson}`
  }

  try {
    return await mcp.callTool(name, args)
  } catch (err) {
    return `Error: ${err instanceof Error ? err.message : String(err)}`
  }
}
