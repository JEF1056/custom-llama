/**
 * Firebase Auth utilities.
 * Pattern mirrors portfolio's src/services/auth.ts.
 */
import {
  GoogleAuthProvider,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  type User,
} from 'firebase/auth'
import { auth, isFirebaseConfigured } from './firebase'

// Comma-separated UIDs from env var (same as portfolio's VITE_ADMIN_UID)
const ALLOWED_UIDS: string[] = (import.meta.env.VITE_ADMIN_UID ?? '')
  .split(',')
  .map((s: string) => s.trim())
  .filter(Boolean)

export const hasAllowedUIDs = ALLOWED_UIDS.length > 0

export function isAllowedUser(user: User | null): boolean {
  if (!user) return false
  if (!hasAllowedUIDs) return false // deny all if no UIDs configured
  return ALLOWED_UIDS.includes(user.uid)
}

function isMobile(): boolean {
  return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
}

export async function signInWithGoogle(): Promise<User | null> {
  if (!auth || !isFirebaseConfigured) throw new Error('Firebase not configured')
  const provider = new GoogleAuthProvider()
  provider.setCustomParameters({ prompt: 'select_account' })

  if (isMobile()) {
    await signInWithRedirect(auth, provider)
    return null // redirect flow — result handled in checkRedirectResult
  }

  const result = await signInWithPopup(auth, provider)
  return result.user
}

export async function checkRedirectResult(): Promise<User | null> {
  if (!auth || !isFirebaseConfigured) return null
  try {
    const result = await getRedirectResult(auth)
    return result?.user ?? null
  } catch {
    return null
  }
}

export async function signOut(): Promise<void> {
  if (!auth) return
  await firebaseSignOut(auth)
}

export function onAuthChange(callback: (user: User | null) => void): () => void {
  if (!auth) {
    callback(null)
    return () => {}
  }
  return onAuthStateChanged(auth, callback)
}

export type { User }
