import { useState } from 'react'
import { ChevronRight, Wrench } from 'lucide-react'
import type { ToolCall, ToolResult } from '../types'

interface Props {
  toolCall: ToolCall
  result?: ToolResult
}

export function ToolCallBlock({ toolCall, result }: Props) {
  const [expanded, setExpanded] = useState(false)

  let argsFormatted = toolCall.function.arguments
  try {
    argsFormatted = JSON.stringify(JSON.parse(toolCall.function.arguments), null, 2)
  } catch {
    // leave as-is
  }

  const isError = result?.isError ?? false

  return (
    <div className="my-1.5 rounded-xl border border-[var(--border)] overflow-hidden text-xs">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-[var(--bg-elevated)] hover:bg-[var(--bg-surface)] transition-colors text-left"
      >
        <Wrench size={12} className="text-[var(--text-secondary)] shrink-0" />
        <span className="font-mono font-medium text-[var(--text-primary)] truncate">
          {toolCall.function.name}
        </span>

        {!result ? (
          <span className="ml-auto flex items-center gap-1 text-[var(--text-secondary)]">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            Running…
          </span>
        ) : isError ? (
          <span className="ml-auto text-red-500">Error</span>
        ) : (
          <span className="ml-auto text-green-500">Done</span>
        )}

        <ChevronRight
          size={12}
          className={`text-[var(--text-secondary)] transition-transform shrink-0 ${expanded ? 'rotate-90' : ''}`}
        />
      </button>

      {expanded && (
        <div className="divide-y divide-[var(--border)]">
          <div className="px-3 py-2 bg-[var(--bg-base)]">
            <div className="text-[10px] uppercase tracking-wide text-[var(--text-secondary)] mb-1">Input</div>
            <pre className="font-mono text-[var(--text-primary)] whitespace-pre-wrap break-all leading-relaxed">
              {argsFormatted}
            </pre>
          </div>
          {result && (
            <div className="px-3 py-2 bg-[var(--bg-base)]">
              <div className="text-[10px] uppercase tracking-wide text-[var(--text-secondary)] mb-1">Output</div>
              <pre
                className={`font-mono whitespace-pre-wrap break-all leading-relaxed max-h-48 overflow-y-auto ${
                  isError ? 'text-red-500' : 'text-[var(--text-primary)]'
                }`}
              >
                {result.content}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
