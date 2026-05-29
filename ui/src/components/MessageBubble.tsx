import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import type { Message, ContentPart } from '../types'
import { ThinkingBlock } from './ThinkingBlock'
import { ToolCallBlock } from './ToolCallBlock'

// Allow code highlighting classes from rehype-highlight
const sanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    code: [...(defaultSchema.attributes?.code ?? []), ['className', /^language-/]],
    span: [...(defaultSchema.attributes?.span ?? []), ['className', /^hljs/]],
  },
}

interface Props {
  message: Message
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const isTool = message.role === 'tool'

  if (isTool) return null // tool results are shown inside ToolCallBlock

  const textContent = extractTextContent(message)

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[85%] ${isUser ? 'max-w-[75%]' : 'w-full max-w-[85%]'}`}>
        {/* Thinking block (assistant only) */}
        {!isUser && (message.thinking || message.streaming) && (
          <ThinkingBlock thinking={message.thinking ?? ''} streaming={message.streaming} />
        )}

        {/* Tool calls (assistant only) */}
        {!isUser && message.tool_calls && message.tool_calls.length > 0 && (
          <div className="space-y-1 mb-2">
            {message.tool_calls.map((tc) => (
              <ToolCallBlock
                key={tc.id}
                toolCall={tc}
                result={message.toolResults?.[tc.id]}
              />
            ))}
          </div>
        )}

        {/* Image attachments (user messages) */}
        {isUser && Array.isArray(message.content) && (
          <div className="flex flex-wrap gap-2 mb-2 justify-end">
            {(message.content as ContentPart[])
              .filter((p): p is Extract<ContentPart, { type: 'image_url' }> => p.type === 'image_url')
              .map((p, i) => (
                <img
                  key={i}
                  src={p.image_url.url}
                  alt={`Attachment ${i + 1}`}
                  className="max-h-48 max-w-full rounded-xl border border-[var(--border)] object-contain"
                />
              ))}
          </div>
        )}

        {/* Message body */}
        {textContent && (
          <div
            className={
              isUser
                ? 'px-4 py-3 rounded-2xl rounded-tr-sm bg-[var(--text-primary)] text-[var(--bg-base)] text-sm whitespace-pre-wrap'
                : 'prose prose-sm dark:prose-invert max-w-none'
            }
          >
            {isUser ? (
              textContent
            ) : (
              <>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight, [rehypeSanitize, sanitizeSchema]]}
                  components={{
                    // Open links in new tab
                    a: ({ href, children }) => (
                      <a href={href} target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    ),
                    // Style code blocks
                    pre: ({ children }) => (
                      <pre className="overflow-x-auto rounded-xl bg-[var(--bg-elevated)] border border-[var(--border)] p-3 text-xs">
                        {children}
                      </pre>
                    ),
                  }}
                >
                  {textContent}
                </ReactMarkdown>

                {/* Streaming cursor */}
                {message.streaming && !message.tool_calls?.length && (
                  <span className="inline-block w-1.5 h-4 bg-current opacity-70 animate-pulse ml-0.5 align-text-bottom" />
                )}
              </>
            )}
          </div>
        )}

        {/* Token speed — live during streaming, dimmed summary after */}
        {!isUser && (message.streaming || !!message.tokensPerSec) && (
          <p className={`mt-1 text-[10px] font-mono tabular-nums ${
            message.streaming
              ? 'text-[var(--text-secondary)]'
              : 'text-[var(--placeholder)] opacity-60'
          }`}>
            {message.streaming && message.tokensPerSec
              ? `⚡ ${message.tokensPerSec} tok/s`
              : message.streaming
              ? '…'
              : `${message.tokenCount ? `${message.tokenCount} tok · ` : ''}${message.tokensPerSec} tok/s`}
          </p>
        )}

        {/* Agent iteration badge */}
        {message.agentIteration && message.agentIteration > 1 && (
          <p className="mt-0.5 text-[10px] text-[var(--text-secondary)]">
            Turn {message.agentIteration}
          </p>
        )}
      </div>
    </div>
  )
}

function extractTextContent(message: Message): string {
  if (typeof message.content === 'string') return message.content
  const parts = message.content as ContentPart[]
  return parts
    .filter((p): p is Extract<ContentPart, { type: 'text' }> => p.type === 'text')
    .map((p) => p.text)
    .join('\n')
}
