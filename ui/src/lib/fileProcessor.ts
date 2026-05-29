/**
 * Central file attachment processor.
 * Dispatches each file to the appropriate pipeline and returns a ProcessedAttachment.
 */
import type { ProcessedAttachment, SheetInfo } from '../types'
import type { MCPClient } from './mcp'
import { pdfToImages } from './pdf'

// ── Constants ─────────────────────────────────────────────────────────────────

const TEXT_EXTENSIONS = new Set([
  'txt', 'md', 'csv', 'yaml', 'yml', 'toml', 'ini', 'log',
  'sh', 'bash', 'zsh', 'py', 'js', 'ts', 'jsx', 'tsx',
  'html', 'htm', 'xml', 'json', 'css', 'scss', 'less',
  'rs', 'go', 'java', 'c', 'cpp', 'h', 'hpp', 'cs',
  'rb', 'php', 'swift', 'kt', 'r', 'sql', 'graphql',
  'conf', 'cfg', 'env', 'gitignore', 'dockerfile',
])

const MAX_TEXT_BYTES = 100 * 1024 // 100 KB

function getExtension(file: File): string {
  const name = file.name
  const dot = name.lastIndexOf('.')
  return dot >= 0 ? name.slice(dot + 1).toLowerCase() : ''
}

function prefixedFilename(file: File): string {
  // Avoid collisions: timestamp_originalname
  // Only strip path separators and double-dots; preserve single dots (extensions)
  const clean = file.name.replace(/[/\\]/g, '_').replace(/\.\.+/g, '_')
  return `${Date.now()}_${clean}`
}

// ── Pipeline: image ───────────────────────────────────────────────────────────

async function processImage(file: File): Promise<ProcessedAttachment> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      const base64 = result.split(',')[1] ?? ''
      resolve({ kind: 'image', file, base64, mimeType: file.type })
    }
    reader.onerror = () => reject(new Error('Failed to read image'))
    reader.readAsDataURL(file)
  })
}

// ── Pipeline: PDF ─────────────────────────────────────────────────────────────

async function processPdf(file: File): Promise<ProcessedAttachment> {
  const { pages, pageCount } = await pdfToImages(file)
  return { kind: 'pdf', file, pages, pageCount }
}

// ── Pipeline: text ────────────────────────────────────────────────────────────

async function processText(file: File): Promise<ProcessedAttachment> {
  const ext = getExtension(file)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    // Slice at MAX_TEXT_BYTES + 1 to detect truncation
    const slice = file.slice(0, MAX_TEXT_BYTES + 1)
    reader.onload = () => {
      let content = reader.result as string
      const truncated = content.length > MAX_TEXT_BYTES
      if (truncated) content = content.slice(0, MAX_TEXT_BYTES)
      const lineCount = content.split('\n').length
      resolve({ kind: 'text', file, content, truncated, ext, lineCount })
    }
    reader.onerror = () => reject(new Error('Failed to read text file'))
    reader.readAsText(slice)
  })
}

// ── Pipeline: Excel (SheetJS lazy-import) ─────────────────────────────────────

const EXCEL_MAX_ROWS = 50
const EXCEL_MAX_COLS = 10

async function processExcel(file: File, mcpClient: MCPClient): Promise<ProcessedAttachment> {
  // Lazy-import SheetJS for bundle size
  const XLSX = await import('xlsx')
  const ab = await file.arrayBuffer()
  const wb = XLSX.read(ab, { type: 'array' })

  const sheetNames = wb.SheetNames
  const firstSheet = wb.Sheets[sheetNames[0]]
  const range = XLSX.utils.decode_range(firstSheet['!ref'] ?? 'A1')
  const rowCount = range.e.r - range.s.r + 1
  const colCount = range.e.c - range.s.c + 1

  const sheetInfo: SheetInfo = { sheetNames, rowCount, colCount }

  // Build markdown table preview (max EXCEL_MAX_ROWS × EXCEL_MAX_COLS)
  const rows: string[][] = XLSX.utils.sheet_to_json(firstSheet, {
    header: 1,
    range: 0,
    defval: '',
  }) as string[][]

  const previewRows = rows.slice(0, EXCEL_MAX_ROWS + 1) // +1 for header
  const previewCols = EXCEL_MAX_COLS

  function truncateRow(row: string[]): string[] {
    const cells = row.slice(0, previewCols).map((c) => String(c ?? '').replace(/\|/g, '\\|'))
    if (colCount > previewCols) cells.push('…')
    return cells
  }

  const mdRows: string[] = []
  if (previewRows.length > 0) {
    mdRows.push(`| ${truncateRow(previewRows[0]).join(' | ')} |`)
    mdRows.push(`| ${truncateRow(previewRows[0]).map(() => '---').join(' | ')} |`)
    for (let i = 1; i < previewRows.length; i++) {
      mdRows.push(`| ${truncateRow(previewRows[i]).join(' | ')} |`)
    }
  }

  const previewTable = mdRows.join('\n')
  const uploadedName = prefixedFilename(file)

  try {
    const { filename, downloadUrl } = await mcpClient.uploadFile(file, uploadedName)
    const preview = buildExcelInjection(file.name, filename, previewTable, sheetInfo, rowCount > EXCEL_MAX_ROWS)
    return { kind: 'excel', file, preview, mcpFilename: filename, downloadUrl, sheetInfo }
  } catch (err) {
    throw new Error(`Excel upload failed: ${err instanceof Error ? err.message : String(err)}`)
  }
}

function buildExcelInjection(
  originalName: string,
  mcpFilename: string,
  previewTable: string,
  sheetInfo: SheetInfo,
  truncated: boolean,
): string {
  const sheetLabel = sheetInfo.sheetNames.join(', ')
  const rows = sheetInfo.rowCount
  const cols = sheetInfo.colCount
  const truncNote = truncated ? `\n(Showing first ${EXCEL_MAX_ROWS} of ${rows} rows — use xlsx_read for full data)` : ''
  return (
    `[File: ${originalName}]\n` +
    `Uploaded to MCP server as: "${mcpFilename}"\n` +
    `Tool access: xlsx_read(filename="${mcpFilename}")\n\n` +
    `Spreadsheet preview — ${sheetLabel} (${rows} rows, ${cols} columns):\n` +
    previewTable +
    truncNote
  )
}

// ── Pipeline: MCP-only (PPTX, DOCX, etc.) ────────────────────────────────────

function getPptxToolHint(mcpFilename: string): string {
  return (
    `pptx_read(filename="${mcpFilename}")\n` +
    `pptx_slide_image(filename="${mcpFilename}", slide_index=0)`
  )
}

function getGenericToolHint(mcpFilename: string): string {
  return `file_read(filename="${mcpFilename}")`
}

async function processMcpFile(file: File, mcpClient: MCPClient): Promise<ProcessedAttachment> {
  const ext = getExtension(file)
  const uploadedName = prefixedFilename(file)

  const { filename, downloadUrl } = await mcpClient.uploadFile(file, uploadedName)

  const toolHint = ['pptx', 'ppt'].includes(ext)
    ? getPptxToolHint(filename)
    : getGenericToolHint(filename)

  return { kind: 'mcp', file, mcpFilename: filename, downloadUrl, toolHint }
}

// ── Central dispatcher ────────────────────────────────────────────────────────

export async function processAttachment(
  file: File,
  mcpClient: MCPClient | null,
): Promise<ProcessedAttachment> {
  try {
    const ext = getExtension(file)
    const mime = file.type

    // Images
    if (mime.startsWith('image/') || ['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) {
      return await processImage(file)
    }

    // PDF
    if (mime === 'application/pdf' || ext === 'pdf') {
      return await processPdf(file)
    }

    // Plain text formats
    if (TEXT_EXTENSIONS.has(ext)) {
      return await processText(file)
    }

    // Excel
    if (['xlsx', 'xls'].includes(ext)) {
      if (!mcpClient) throw new Error('MCP not configured — cannot upload Excel file')
      return await processExcel(file, mcpClient)
    }

    // Everything else — upload to MCP
    if (!mcpClient) throw new Error('MCP not configured — cannot upload binary file')
    return await processMcpFile(file, mcpClient)
  } catch (err) {
    return {
      kind: 'error',
      file,
      error: err instanceof Error ? err.message : String(err),
    }
  }
}

// ── Content injection builder ─────────────────────────────────────────────────

/**
 * Build the text string to prepend to the user message for a non-image attachment.
 * Images are sent as image_url content parts; everything else needs a text injection.
 */
export function buildAttachmentInjection(att: ProcessedAttachment): string | null {
  switch (att.kind) {
    case 'image':
      return null // handled as image_url content part

    case 'pdf':
      return null // pages sent as image_url content parts

    case 'text': {
      const lang = att.ext || 'text'
      const truncNote = att.truncated ? '\n…[truncated at 100KB]' : ''
      return `\`\`\`${lang}\n${att.content}${truncNote}\n\`\`\``
    }

    case 'excel':
      return att.preview

    case 'mcp': {
      const { file, mcpFilename, toolHint } = att
      return (
        `[File: ${file.name}]\n` +
        `Uploaded to MCP server as: "${mcpFilename}"\n` +
        `Tool access:\n${toolHint}`
      )
    }

    case 'error':
      return `[File: ${att.file.name}]\nError processing file: ${att.error}`
  }
}
