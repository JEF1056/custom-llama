import { useState } from 'react'
import { Plus, Trash2, MessageSquare, Check, Pencil } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import type { Conversation } from '../types'
import { signOut } from '../lib/auth'
import { isFirebaseConfigured } from '../lib/firebase'

interface Props {
  user?: { displayName: string | null; email: string | null; photoURL: string | null } | null
}

function groupByDate(convs: Conversation[]): { label: string; items: Conversation[] }[] {
  const now = Date.now()
  const MS_DAY = 86400000

  const groups: Record<string, Conversation[]> = {
    Today: [],
    Yesterday: [],
    'This Week': [],
    Older: [],
  }

  for (const c of convs) {
    const age = now - c.updatedAt
    if (age < MS_DAY) groups['Today'].push(c)
    else if (age < 2 * MS_DAY) groups['Yesterday'].push(c)
    else if (age < 7 * MS_DAY) groups['This Week'].push(c)
    else groups['Older'].push(c)
  }

  return Object.entries(groups)
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }))
}

export function Sidebar({ user }: Props) {
  const { conversations, activeId, createConversation, deleteConversation, renameConversation, setActiveId } =
    useChatStore()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')

  const groups = groupByDate(conversations)

  function handleNewChat() {
    createConversation()
  }

  function startEdit(conv: Conversation) {
    setEditingId(conv.id)
    setEditValue(conv.title)
  }

  function commitEdit(id: string) {
    renameConversation(id, editValue)
    setEditingId(null)
  }

  return (
    <aside className="w-64 shrink-0 flex flex-col h-full bg-[var(--bg-surface)] border-r border-[var(--border)]">
      {/* New chat */}
      <div className="p-3">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-[var(--border)] text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors"
        >
          <Plus size={15} />
          New chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-4">
        {groups.length === 0 && (
          <p className="text-xs text-[var(--text-secondary)] text-center py-6 px-3">
            No conversations yet. Start by typing a message.
          </p>
        )}

        {groups.map(({ label, items }) => (
          <div key={label}>
            <p className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
              {label}
            </p>
            <div className="space-y-0.5">
              {items.map((conv) => (
                <ConvItem
                  key={conv.id}
                  conv={conv}
                  active={conv.id === activeId}
                  editing={editingId === conv.id}
                  editValue={editValue}
                  onSelect={() => setActiveId(conv.id)}
                  onDelete={() => deleteConversation(conv.id)}
                  onStartEdit={() => startEdit(conv)}
                  onEditChange={setEditValue}
                  onEditCommit={() => commitEdit(conv.id)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      {isFirebaseConfigured && user && (
        <div className="p-3 border-t border-[var(--border)] flex items-center gap-2">
          {user.photoURL ? (
            <img src={user.photoURL} alt="" className="w-7 h-7 rounded-full shrink-0" referrerPolicy="no-referrer" />
          ) : (
            <div className="w-7 h-7 rounded-full bg-[var(--bg-elevated)] flex items-center justify-center text-xs font-semibold text-[var(--text-secondary)] shrink-0">
              {(user.displayName ?? user.email ?? '?')[0].toUpperCase()}
            </div>
          )}
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-[var(--text-primary)] truncate">
              {user.displayName ?? user.email ?? 'User'}
            </p>
          </div>
          <button
            onClick={signOut}
            title="Sign out"
            className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] p-1 rounded-lg hover:bg-[var(--bg-elevated)] transition-colors"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
          </button>
        </div>
      )}
    </aside>
  )
}

// ── ConvItem ──────────────────────────────────────────────────────────────────

interface ConvItemProps {
  conv: Conversation
  active: boolean
  editing: boolean
  editValue: string
  onSelect: () => void
  onDelete: () => void
  onStartEdit: () => void
  onEditChange: (v: string) => void
  onEditCommit: () => void
}

function ConvItem({
  conv,
  active,
  editing,
  editValue,
  onSelect,
  onDelete,
  onStartEdit,
  onEditChange,
  onEditCommit,
}: ConvItemProps) {
  return (
    <div
      className={`group relative flex items-center rounded-xl px-3 py-2 cursor-pointer transition-colors ${
        active ? 'bg-[var(--bg-elevated)]' : 'hover:bg-[var(--bg-elevated)]'
      }`}
      onClick={onSelect}
    >
      <MessageSquare size={13} className="shrink-0 text-[var(--text-secondary)] mr-2" />

      {editing ? (
        <input
          autoFocus
          value={editValue}
          onChange={(e) => onEditChange(e.target.value)}
          onBlur={onEditCommit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onEditCommit()
            if (e.key === 'Escape') onEditCommit()
            e.stopPropagation()
          }}
          onClick={(e) => e.stopPropagation()}
          className="flex-1 min-w-0 bg-transparent text-sm text-[var(--text-primary)] outline-none border-b border-[var(--border)]"
        />
      ) : (
        <span className="flex-1 min-w-0 text-sm text-[var(--text-primary)] truncate">{conv.title}</span>
      )}

      <div
        className="hidden group-hover:flex items-center gap-1 ml-1 shrink-0"
        onClick={(e) => e.stopPropagation()}
      >
        {editing ? (
          <button onClick={onEditCommit} className="p-1 text-green-500 hover:text-green-600">
            <Check size={12} />
          </button>
        ) : (
          <button
            onClick={onStartEdit}
            className="p-1 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            title="Rename"
          >
            <Pencil size={12} />
          </button>
        )}
        <button
          onClick={onDelete}
          className="p-1 text-[var(--text-secondary)] hover:text-red-500"
          title="Delete"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  )
}
