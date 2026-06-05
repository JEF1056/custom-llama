import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Conversation, Message, ToolResult } from '../types'

const MAX_CONVERSATIONS = 100

function generateId() {
  return crypto.randomUUID()
}

function makeTitle(text: string): string {
  // Auto-title from first user message text content
  const clean = text
    .replace(/\[File:.*?\]/gs, '')
    .replace(/```[\s\S]*?```/g, '')
    .trim()
    .slice(0, 60)
    .trim()
  return clean || 'New conversation'
}

interface ChatStore {
  conversations: Conversation[]
  activeId: string | null

  // Conversation ops
  createConversation: () => string
  deleteConversation: (id: string) => void
  renameConversation: (id: string, title: string) => void
  clearMessages: (id: string) => void
  setActiveId: (id: string | null) => void

  // Message ops
  appendMessage: (convId: string, msg: Omit<Message, 'id' | 'timestamp'>) => string
  updateMessage: (convId: string, msgId: string, patch: Partial<Message>) => void
  appendToolResult: (convId: string, toolCallId: string, result: ToolResult) => void

  // Helpers
  activeConversation: () => Conversation | null
  getConversation: (id: string) => Conversation | undefined
  /** Clear streaming flags left over from interrupted sessions (call on app mount) */
  clearStreamingFlags: () => void
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeId: null,

      createConversation: () => {
        const id = generateId()
        const conv: Conversation = {
          id,
          title: 'New conversation',
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [],
        }
        set((state) => {
          const convs = [conv, ...state.conversations].slice(0, MAX_CONVERSATIONS)
          return { conversations: convs, activeId: id }
        })
        return id
      },

      deleteConversation: (id) =>
        set((state) => {
          const convs = state.conversations.filter((c) => c.id !== id)
          const activeId = state.activeId === id
            ? (convs[0]?.id ?? null)
            : state.activeId
          return { conversations: convs, activeId }
        }),

      renameConversation: (id, title) =>
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, title: title.trim() || 'New conversation', updatedAt: Date.now() } : c,
          ),
        })),

      clearMessages: (id) =>
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, messages: [], updatedAt: Date.now() } : c,
          ),
        })),

      setActiveId: (id) => set({ activeId: id }),

      appendMessage: (convId, msgData) => {
        const id = generateId()
        const msg: Message = { ...msgData, id, timestamp: Date.now() }
        set((state) => ({
          conversations: state.conversations.map((c) => {
            if (c.id !== convId) return c
            const messages = [...c.messages, msg]
            // Auto-title from first user message
            let title = c.title
            if (title === 'New conversation' && msg.role === 'user') {
              const text = typeof msg.content === 'string'
                ? msg.content
                : msg.content.filter((p) => p.type === 'text').map((p) => (p as { text: string }).text).join(' ')
              title = makeTitle(text)
            }
            return { ...c, messages, title, updatedAt: Date.now() }
          }),
        }))
        return id
      },

      updateMessage: (convId, msgId, patch) =>
        set((state) => ({
          conversations: state.conversations.map((c) => {
            if (c.id !== convId) return c
            return {
              ...c,
              updatedAt: Date.now(),
              messages: c.messages.map((m) => (m.id === msgId ? { ...m, ...patch } : m)),
            }
          }),
        })),

      appendToolResult: (convId, toolCallId, result) =>
        set((state) => ({
          conversations: state.conversations.map((c) => {
            if (c.id !== convId) return c
            // Find the last assistant message with the matching tool call
            const messages = c.messages.map((m) => {
              if (m.role !== 'assistant' || !m.tool_calls?.find((tc) => tc.id === toolCallId)) return m
              return {
                ...m,
                toolResults: { ...(m.toolResults ?? {}), [toolCallId]: result },
              }
            })
            return { ...c, messages, updatedAt: Date.now() }
          }),
        })),

      activeConversation: () => {
        const { conversations, activeId } = get()
        return conversations.find((c) => c.id === activeId) ?? null
      },

      getConversation: (id) => get().conversations.find((c) => c.id === id),

      clearStreamingFlags: () =>
        set((state) => ({
          conversations: state.conversations.map((c) => ({
            ...c,
            messages: c.messages.map((m) =>
              m.streaming ? { ...m, streaming: false } : m,
            ),
          })),
        })),
    }),
    {
      name: 'chat-history',
      partialize: (state) => ({
        conversations: state.conversations,
        activeId: state.activeId,
      }),
    },
  ),
)
