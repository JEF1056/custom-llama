import { useEffect, useState } from 'react'
import { Settings, PanelLeftClose, PanelLeft, MessageSquarePlus } from 'lucide-react'
import { AuthGate } from './components/AuthGate'
import { Sidebar } from './components/Sidebar'
import { ChatWindow } from './components/ChatWindow'
import { SettingsModal } from './components/SettingsModal'
import { ThemeToggle } from './components/ThemeToggle'
import { onAuthChange, type User } from './lib/auth'
import { isFirebaseConfigured } from './lib/firebase'
import { useChatStore } from './store/chatStore'
import { useSettingsStore } from './store/settingsStore'

export default function App() {
  const [user, setUser] = useState<User | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const { settings, updateSettings } = useSettingsStore()
  const { createConversation, conversations, activeId, setActiveId } = useChatStore()

  useEffect(() => {
    // Clear any streaming flags that survived a page refresh
    useChatStore.getState().clearStreamingFlags()
  }, [])

  useEffect(() => {
    if (!isFirebaseConfigured) return
    return onAuthChange(setUser)
  }, [])

  // Ensure there's always an active conversation
  useEffect(() => {
    if (conversations.length === 0) {
      createConversation()
    } else if (!activeId) {
      setActiveId(conversations[0].id)
    }
  }, [conversations.length, activeId])

  const sidebarOpen = settings.sidebarOpen

  return (
    <AuthGate>
      <div className="flex flex-col h-screen bg-[var(--bg-base)] overflow-hidden">
        {/* Header */}
        <header className="h-12 shrink-0 flex items-center gap-2 px-3 border-b border-[var(--border)] bg-[var(--bg-surface)]">
          <button
            onClick={() => updateSettings({ sidebarOpen: !sidebarOpen })}
            className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors"
            title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            {sidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeft size={16} />}
          </button>

          <span className="text-sm font-semibold text-[var(--text-primary)] mr-auto">Chat</span>

          <button
            onClick={createConversation}
            className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors"
            title="New chat"
          >
            <MessageSquarePlus size={16} />
          </button>

          <ThemeToggle />

          <button
            onClick={() => setSettingsOpen(true)}
            className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors"
            title="Settings"
          >
            <Settings size={16} />
          </button>
        </header>

        {/* Body */}
        <div className="flex flex-1 min-h-0">
          {sidebarOpen && (
            <Sidebar user={isFirebaseConfigured ? user : undefined} />
          )}
          <main className="flex-1 flex flex-col min-w-0">
            <ChatWindow />
          </main>
        </div>
      </div>

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </AuthGate>
  )
}
