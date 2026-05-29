import { useState } from 'react'
import { ChevronRight } from 'lucide-react'

interface Props {
  thinking: string
  streaming?: boolean
}

export function ThinkingBlock({ thinking, streaming }: Props) {
  const [expanded, setExpanded] = useState(false)

  if (!thinking && !streaming) return null

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
      >
        <ChevronRight
          size={13}
          className={`transition-transform ${expanded ? 'rotate-90' : ''}`}
        />
        {streaming && !thinking ? (
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            Thinking…
          </span>
        ) : (
          <span>Thoughts {expanded ? '▾' : '▸'}</span>
        )}
      </button>

      {expanded && thinking && (
        <div className="mt-2 pl-4 border-l-2 border-[var(--border)] text-xs text-[var(--text-secondary)] whitespace-pre-wrap font-mono leading-relaxed max-h-64 overflow-y-auto">
          {thinking}
        </div>
      )}
    </div>
  )
}
