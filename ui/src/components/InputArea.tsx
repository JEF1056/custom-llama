import { useState, useRef, type KeyboardEvent, type DragEvent } from 'react'
import { Paperclip, Send, Square } from 'lucide-react'
import type { ProcessedAttachment } from '../types'
import { processAttachment } from '../lib/fileProcessor'
import { getMcpClient } from '../lib/mcp'
import { useSettingsStore } from '../store/settingsStore'
import { FileBadge } from './FileBadge'

type UploadStatus = 'processing' | 'uploading' | 'done' | 'error'

interface AttachmentEntry {
  att: ProcessedAttachment
  status: UploadStatus
}

interface Props {
  onSend: (text: string, attachments: ProcessedAttachment[]) => void
  disabled?: boolean
  onAbort?: () => void
}

export function InputArea({ onSend, disabled, onAbort }: Props) {
  const { settings } = useSettingsStore()
  const [text, setText] = useState('')
  const [entries, setEntries] = useState<AttachmentEntry[]>([])
  const [dragging, setDragging] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function canSend() {
    return !disabled && (text.trim().length > 0 || entries.some((e) => e.status === 'done'))
  }

  async function processFiles(files: File[]) {
    const mcpClient = settings.enableMcp
      ? getMcpClient(settings.mcpBaseUrl, settings.mcpApiKey)
      : null

    for (const file of files) {
      // Add placeholder
      const placeholder: AttachmentEntry = {
        att: { kind: 'error', file, error: '' }, // will be replaced
        status: 'processing',
      }
      setEntries((prev) => [...prev, placeholder])

      const result = await processAttachment(file, mcpClient)

      setEntries((prev) => {
        const next = [...prev]
        // Replace the last 'processing' entry for this file (ES2020-safe reverse scan)
        let i = next.length - 1
        while (i >= 0 && !(next[i].status === 'processing' && next[i].att.file === file)) i--
        if (i >= 0) {
          next[i] = {
            att: result,
            status: result.kind === 'error' ? 'error' : 'done',
          }
        }
        return next
      })
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    if (files.length) processFiles(files)
    e.target.value = ''
  }

  function removeEntry(i: number) {
    setEntries((prev) => prev.filter((_, j) => j !== i))
  }

  function handleSend() {
    if (!canSend()) return
    const readyAtts = entries.filter((e) => e.status === 'done').map((e) => e.att)
    onSend(text.trim(), readyAtts)
    setText('')
    setEntries([])
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Auto-resize textarea
  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setText(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }

  // Drag-and-drop
  function onDragOver(e: DragEvent) {
    e.preventDefault()
    setDragging(true)
  }
  function onDragLeave() {
    setDragging(false)
  }
  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length) processFiles(files)
  }

  return (
    <div
      className={`p-3 border-t border-[var(--border)] bg-[var(--bg-surface)] transition-colors ${
        dragging ? 'bg-[var(--bg-elevated)]' : ''
      }`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {/* File badges */}
      {entries.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2 px-1">
          {entries.map((entry, i) => (
            <FileBadge
              key={i}
              attachment={entry.att}
              status={entry.status}
              onRemove={() => removeEntry(i)}
            />
          ))}
        </div>
      )}

      {/* Input row */}
      <div className="flex items-end gap-2">
        {/* Attach */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="p-2.5 rounded-xl text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors disabled:opacity-40 shrink-0"
          title="Attach file"
        >
          <Paperclip size={17} />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={handleFileChange}
        />

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={dragging ? 'Drop files here…' : 'Message…'}
          rows={1}
          disabled={disabled}
          className="flex-1 resize-none px-4 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--bg-base)] text-sm text-[var(--text-primary)] placeholder:text-[var(--placeholder)] focus:outline-none focus:ring-2 focus:ring-black/10 dark:focus:ring-white/10 disabled:opacity-50 leading-relaxed"
          style={{ minHeight: '42px', maxHeight: '200px' }}
        />

        {/* Send / Abort */}
        {disabled ? (
          <button
            onClick={onAbort}
            className="p-2.5 rounded-xl bg-[var(--text-primary)] text-[var(--bg-base)] hover:opacity-80 transition-opacity shrink-0"
            title="Stop"
          >
            <Square size={15} fill="currentColor" />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!canSend()}
            className="p-2.5 rounded-xl bg-[var(--text-primary)] text-[var(--bg-base)] hover:opacity-80 transition-opacity disabled:opacity-30 shrink-0"
            title="Send (Enter)"
          >
            <Send size={15} />
          </button>
        )}
      </div>

      <p className="text-[10px] text-[var(--text-secondary)] text-center mt-2">
        Shift+Enter for newline · Drop files to attach
      </p>
    </div>
  )
}
