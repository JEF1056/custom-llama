import { useEffect, useRef } from 'react'
import { Bot } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import { useChat } from '../hooks/useChat'
import { useMCP } from '../hooks/useMCP'
import { MessageBubble } from './MessageBubble'
import { InputArea } from './InputArea'
import type { ProcessedAttachment } from '../types'

export function ChatWindow() {
  const { activeConversation } = useChatStore()
  const conv = activeConversation()

  const mcp = useMCP()
  const { isStreaming, agentIteration, sendMessage, abort } = useChat(conv?.id ?? null, mcp)

  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conv?.messages.length])

  async function handleSend(text: string, attachments: ProcessedAttachment[]) {
    await sendMessage(text, attachments)
  }

  // Empty state
  if (!conv) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-4">
        <div className="w-12 h-12 rounded-2xl bg-[var(--bg-elevated)] flex items-center justify-center">
          <Bot size={22} className="text-[var(--text-secondary)]" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-[var(--text-primary)]">Start a conversation</h2>
          <p className="text-sm text-[var(--text-secondary)] mt-1">Type a message below to begin</p>
        </div>
      </div>
    )
  }

  const messages = conv.messages.filter((m) => m.role !== 'tool')

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto">
          {messages.length === 0 && (
            <div className="text-center text-sm text-[var(--text-secondary)] py-12">
              No messages yet. Say hello!
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Agent status */}
          {isStreaming && agentIteration && agentIteration > 0 && (
            <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] mb-4">
              <div className="flex gap-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              Turn {agentIteration}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* MCP error banner */}
      {mcp.error && (
        <div className="mx-4 mb-2 px-3 py-2 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-xs text-amber-700 dark:text-amber-300 flex items-center justify-between gap-2">
          <span>MCP: {mcp.error}</span>
          <button onClick={mcp.reload} className="underline shrink-0">Retry</button>
        </div>
      )}

      {/* Input */}
      <InputArea
        onSend={handleSend}
        disabled={isStreaming}
        onAbort={abort}
      />
    </div>
  )
}
