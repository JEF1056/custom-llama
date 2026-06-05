import { useEffect, useState, type ReactNode } from 'react'
import type { User } from '../lib/auth'
import {
  signInWithGoogle,
  checkRedirectResult,
  signOut,
  onAuthChange,
  isAllowedUser,
  hasAllowedUIDs,
} from '../lib/auth'
import { isFirebaseConfigured } from '../lib/firebase'
import { ThemeToggle } from './ThemeToggle'

interface Props {
  children: ReactNode
}

type AuthState = 'loading' | 'unauthenticated' | 'denied' | 'allowed'

export function AuthGate({ children }: Props) {
  const [user, setUser] = useState<User | null>(null)
  const [authState, setAuthState] = useState<AuthState>('loading')
  const [signingIn, setSigningIn] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Check redirect result first (mobile flow)
    checkRedirectResult().then((u) => {
      if (u) setUser(u)
    })

    if (!isFirebaseConfigured) {
      setAuthState('allowed')
      return
    }

    return onAuthChange((u) => {
      setUser(u)
      if (!u) {
        setAuthState('unauthenticated')
      } else if (isAllowedUser(u)) {
        setAuthState('allowed')
      } else {
        setAuthState('denied')
      }
    })
  }, [])

  async function handleSignIn() {
    setSigningIn(true)
    setError(null)
    try {
      await signInWithGoogle()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in failed')
    } finally {
      setSigningIn(false)
    }
  }

  async function handleSignOut() {
    await signOut()
  }

  if (!isFirebaseConfigured || authState === 'allowed') {
    return <>{children}</>
  }

  // Full-page centering shell
  const shell = (content: ReactNode) => (
    <div className="min-h-screen flex flex-col bg-[var(--bg-base)]">
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>
      <div className="flex-1 flex items-center justify-center p-6">
        {content}
      </div>
    </div>
  )

  if (authState === 'loading') {
    return shell(
      <div className="flex flex-col items-center gap-3 text-[var(--text-secondary)]">
        <div className="w-8 h-8 border-2 border-current border-t-transparent rounded-full animate-spin" />
        <span className="text-sm">Loading…</span>
      </div>,
    )
  }

  if (authState === 'denied' && user) {
    return shell(
      <div className="bg-[var(--bg-surface)] border border-[var(--border)] rounded-3xl p-8 shadow-[0_20px_60px_rgba(20,20,20,0.12)] w-full max-w-sm text-center space-y-4">
        <div className="text-4xl">🚫</div>
        <h1 className="text-xl font-semibold text-[var(--text-primary)]">Access Denied</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          Your account (<span className="font-mono text-xs">{user.email}</span>) is not authorized.
        </p>
        <p className="text-xs text-[var(--text-secondary)] font-mono bg-[var(--bg-elevated)] rounded-lg px-3 py-2">
          UID: {user.uid}
        </p>
        <button
          onClick={handleSignOut}
          className="w-full px-4 py-2.5 rounded-xl border border-[var(--border)] text-sm text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] transition-colors"
        >
          Sign out
        </button>
      </div>,
    )
  }

  // Unauthenticated
  return shell(
    <div className="bg-[var(--bg-surface)] border border-[var(--border)] rounded-3xl p-8 shadow-[0_20px_60px_rgba(20,20,20,0.12)] w-full max-w-sm space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Chat</h1>
        <p className="text-sm text-[var(--text-secondary)]">Sign in to continue</p>
      </div>

      {error && (
        <p className="text-sm text-red-500 bg-red-50 dark:bg-red-900/20 rounded-xl px-3 py-2">
          {error}
        </p>
      )}

      <button
        onClick={handleSignIn}
        disabled={signingIn}
        className="w-full flex items-center justify-center gap-3 px-4 py-3 rounded-xl bg-[var(--text-primary)] text-[var(--bg-base)] font-medium text-sm hover:opacity-90 transition-opacity disabled:opacity-50"
      >
        {signingIn ? (
          <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : (
          <GoogleIcon />
        )}
        {signingIn ? 'Signing in…' : 'Continue with Google'}
      </button>

      {!hasAllowedUIDs && (
        <p className="text-xs text-[var(--text-secondary)] text-center">
          No authorized users configured (VITE_ADMIN_UID not set)
        </p>
      )}
    </div>,
  )
}

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M17.64 9.205c0-.639-.057-1.252-.164-1.841H9v3.481h4.844a4.14 4.14 0 01-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 009 18z" fill="#34A853"/>
      <path d="M3.964 10.71A5.41 5.41 0 013.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 000 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 00.957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
    </svg>
  )
}
