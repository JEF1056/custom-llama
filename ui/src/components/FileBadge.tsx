import { FileText, Image, FileSpreadsheet, Presentation, File, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import type { ProcessedAttachment } from '../types'

type UploadStatus = 'processing' | 'uploading' | 'done' | 'error'

interface Props {
  attachment: ProcessedAttachment
  status: UploadStatus
  onRemove?: () => void
}

function getIcon(att: ProcessedAttachment) {
  switch (att.kind) {
    case 'image': return <Image size={12} />
    case 'pdf': return <FileText size={12} />
    case 'excel': return <FileSpreadsheet size={12} />
    case 'mcp': {
      const ext = att.file.name.split('.').pop()?.toLowerCase() ?? ''
      if (['pptx', 'ppt'].includes(ext)) return <Presentation size={12} />
      return <File size={12} />
    }
    case 'text': return <FileText size={12} />
    case 'error': return <AlertCircle size={12} className="text-red-500" />
  }
}

function getLabel(att: ProcessedAttachment, status: UploadStatus): string {
  switch (att.kind) {
    case 'image': return `🖼 ${att.file.name}`
    case 'pdf': return `📄 ${att.pageCount} page${att.pageCount !== 1 ? 's' : ''} · ${att.file.name}`
    case 'text': return `📝 ${att.ext} · ${att.lineCount} lines${att.truncated ? ' (truncated)' : ''}`
    case 'excel': {
      const { sheetNames, rowCount, colCount } = att.sheetInfo
      const base = `📊 ${sheetNames.length} sheet${sheetNames.length !== 1 ? 's' : ''} · ${rowCount} rows · ${colCount} cols`
      if (status === 'done') return `${base} → MCP ✓`
      if (status === 'uploading') return `${base} → Uploading…`
      return base
    }
    case 'mcp': {
      const ext = att.file.name.split('.').pop()?.toLowerCase() ?? 'file'
      const icon = ['pptx', 'ppt'].includes(ext) ? '📑' : '📁'
      if (status === 'done') return `${icon} ${ext} → MCP ✓`
      if (status === 'uploading') return `${icon} ${ext} → Uploading…`
      return `${icon} ${att.file.name}`
    }
    case 'error': return `❌ ${att.file.name}: ${att.error}`
  }
}

export function FileBadge({ attachment, status, onRemove }: Props) {
  const isError = attachment.kind === 'error' || status === 'error'

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border max-w-[240px] ${
        isError
          ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-600 dark:text-red-400'
          : 'bg-[var(--bg-elevated)] border-[var(--border)] text-[var(--text-secondary)]'
      }`}
    >
      {status === 'processing' || status === 'uploading' ? (
        <Loader2 size={11} className="shrink-0 animate-spin" />
      ) : status === 'done' ? (
        <CheckCircle2 size={11} className="shrink-0 text-green-500" />
      ) : (
        <span className="shrink-0">{getIcon(attachment)}</span>
      )}

      <span className="truncate">{getLabel(attachment, status)}</span>

      {onRemove && (
        <button
          onClick={onRemove}
          className="ml-0.5 shrink-0 opacity-50 hover:opacity-100 transition-opacity"
          title="Remove"
        >
          ×
        </button>
      )}
    </div>
  )
}
