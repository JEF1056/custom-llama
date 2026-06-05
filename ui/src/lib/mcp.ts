/**
 * MCP Streamable HTTP client (JSON-RPC 2.0).
 * Connects to the MCP server at POST / (root path).
 */
import type { MCPTool, OpenAITool } from '../types'

// ── JSON-RPC helpers ─────────────────────────────────────────────────────────

let _rpcId = 1
function nextId() { return _rpcId++ }

interface JsonRpcRequest {
  jsonrpc: '2.0'
  id: number
  method: string
  params?: unknown
}

interface JsonRpcResponse<T = unknown> {
  jsonrpc: '2.0'
  id: number
  result?: T
  error?: { code: number; message: string; data?: unknown }
}

// ── MCPClient ────────────────────────────────────────────────────────────────

export class MCPClient {
  private baseUrl: string
  private apiKey: string

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '')
    this.apiKey = apiKey
  }

  private headers(): HeadersInit {
    const h: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json, text/event-stream',
    }
    if (this.apiKey) h['Authorization'] = `Bearer ${this.apiKey}`
    return h
  }

  private async rpc<T>(method: string, params?: unknown): Promise<T> {
    const body: JsonRpcRequest = {
      jsonrpc: '2.0',
      id: nextId(),
      method,
      ...(params !== undefined ? { params } : {}),
    }

    const res = await fetch(`${this.baseUrl}/`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(body),
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`MCP request failed: ${res.status} ${text}`)
    }

    const contentType = res.headers.get('content-type') ?? ''

    // Streamable HTTP: may return SSE stream for the response
    if (contentType.includes('text/event-stream')) {
      return await this.readSseResponse<T>(res)
    }

    const json: JsonRpcResponse<T> = await res.json()
    if (json.error) {
      throw new Error(`MCP error ${json.error.code}: ${json.error.message}`)
    }
    return json.result as T
  }

  private async readSseResponse<T>(res: Response): Promise<T> {
    const text = await res.text()
    // Find the last JSON event data in the SSE stream
    const lines = text.split('\n')
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim()
      if (line.startsWith('data: ')) {
        try {
          const json: JsonRpcResponse<T> = JSON.parse(line.slice(6))
          if (json.error) throw new Error(`MCP error ${json.error.code}: ${json.error.message}`)
          return json.result as T
        } catch (e) {
          if (e instanceof Error && e.message.startsWith('MCP error')) throw e
        }
      }
    }
    throw new Error('No valid JSON-RPC response in SSE stream')
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async initialize(): Promise<void> {
    await this.rpc('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: { tools: {} },
      clientInfo: { name: 'chat-ui', version: '1.0.0' },
    })
  }

  // ── Tools ──────────────────────────────────────────────────────────────────

  async listTools(): Promise<MCPTool[]> {
    const result = await this.rpc<{ tools: MCPTool[] }>('tools/list')
    return result.tools ?? []
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<string> {
    const result = await this.rpc<{ content: Array<{ type: string; text?: string }> }>(
      'tools/call',
      { name, arguments: args },
    )
    // Concatenate all text content parts
    return (result.content ?? [])
      .filter((c) => c.type === 'text' && c.text)
      .map((c) => c.text!)
      .join('\n')
  }

  // ── File upload ───────────────────────────────────────────────────────────

  async uploadFile(
    file: File,
    prefixedName: string,
  ): Promise<{ filename: string; downloadUrl: string }> {
    const form = new FormData()
    form.append('file', file, prefixedName)

    const headers: Record<string, string> = {}
    if (this.apiKey) headers['Authorization'] = `Bearer ${this.apiKey}`

    const res = await fetch(`${this.baseUrl}/api/files/upload`, {
      method: 'POST',
      headers,
      body: form,
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`MCP upload failed: ${res.status} ${text}`)
    }

    const json = await res.json()
    return {
      filename: json.filename as string,
      downloadUrl: (json.download_url ?? '') as string,
    }
  }
}

// ── Tool schema conversion ────────────────────────────────────────────────────

export function toOpenAITool(mcpTool: MCPTool): OpenAITool {
  return {
    type: 'function',
    function: {
      name: mcpTool.name,
      description: mcpTool.description,
      strict: true,
      parameters: {
        ...mcpTool.inputSchema,
        additionalProperties: false,
      },
    },
  }
}

// ── Singleton factory ─────────────────────────────────────────────────────────

let _client: MCPClient | null = null
let _clientKey = ''

export function getMcpClient(baseUrl: string, apiKey: string): MCPClient {
  const key = `${baseUrl}::${apiKey}`
  if (!_client || _clientKey !== key) {
    _client = new MCPClient(baseUrl, apiKey)
    _clientKey = key
  }
  return _client
}
