import { useEffect } from 'react'
import { Sun, Moon, Monitor } from 'lucide-react'
import { useSettingsStore } from '../store/settingsStore'

type Theme = 'light' | 'dark' | 'system'

function applyTheme(theme: Theme) {
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  const dark = theme === 'dark' || (theme === 'system' && prefersDark)
  document.documentElement.classList.toggle('dark', dark)
}

export function ThemeToggle() {
  const { settings, updateSettings } = useSettingsStore()
  const theme = settings.theme

  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  // Also react to system pref changes when theme === 'system'
  useEffect(() => {
    if (theme !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = () => applyTheme('system')
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [theme])

  function cycle() {
    const next: Theme = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light'
    updateSettings({ theme: next })
  }

  const icons: Record<Theme, React.ReactNode> = {
    light: <Sun size={16} />,
    dark: <Moon size={16} />,
    system: <Monitor size={16} />,
  }

  const labels: Record<Theme, string> = {
    light: 'Light',
    dark: 'Dark',
    system: 'System',
  }

  return (
    <button
      onClick={cycle}
      title={`Theme: ${labels[theme]}`}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors"
    >
      {icons[theme]}
      <span className="hidden sm:inline">{labels[theme]}</span>
    </button>
  )
}
