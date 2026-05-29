// ── Content parts (OpenAI multimodal format) ──────────────────────────
export interface TextPart {
  type: 'text'
  text: string
}

export interface ImageUrlPart {
  type: 'image_url'
  image_url: { url: string; detail?: 'auto' | 'low' | 'high' }
}

export type ContentPart = TextPart | ImageUrlPart

// ── Tool call / result types ──────────────────────────────────────────
export interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string // JSON string
  }
}

export interface ToolResult {
  toolCallId: string
  content: string
  isError: boolean
}

// ── Messages ──────────────────────────────────────────────────────────
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool'

export interface Message {
  id: string
  role: MessageRole
  /** String for simple text; ContentPart[] for multimodal user messages */
  content: string | ContentPart[]
  /** Tool calls emitted by the assistant */
  tool_calls?: ToolCall[]
  /** For role=tool: the tool call this is a response to */
  tool_call_id?: string
  /** Extracted <think>…</think> block. Stored here, NEVER re-sent to API */
  thinking?: string
  /** Resolved tool results keyed by tool_call_id */
  toolResults?: Record<string, ToolResult>
  timestamp: number
  /** True while the assistant is still streaming */
  streaming?: boolean
  /** Current agentic iteration for assistant messages in a loop */
  agentIteration?: number
}

// ── Conversation ──────────────────────────────────────────────────────
export interface Conversation {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: Message[]
}

// ── File attachments ──────────────────────────────────────────────────
export interface SheetInfo {
  sheetNames: string[]
  rowCount: number
  colCount: number
}

export type ProcessedAttachment =
  | { kind: 'image'; file: File; base64: string; mimeType: string }
  | { kind: 'pdf'; file: File; pages: string[]; pageCount: number }
  | { kind: 'text'; file: File; content: string; truncated: boolean; ext: string; lineCount: number }
  | { kind: 'excel'; file: File; preview: string; mcpFilename: string; downloadUrl: string; sheetInfo: SheetInfo }
  | { kind: 'mcp'; file: File; mcpFilename: string; downloadUrl: string; toolHint: string }
  | { kind: 'error'; file: File; error: string }

// ── MCP tool definitions ──────────────────────────────────────────────
export interface MCPToolParameter {
  type: string
  description?: string
  enum?: string[]
  items?: MCPToolParameter
  properties?: Record<string, MCPToolParameter>
  required?: string[]
}

export interface MCPTool {
  name: string
  description: string
  inputSchema: {
    type: 'object'
    properties?: Record<string, MCPToolParameter>
    required?: string[]
    additionalProperties?: boolean
  }
}

export interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description: string
    strict: boolean
    parameters: {
      type: 'object'
      properties?: Record<string, MCPToolParameter>
      required?: string[]
      additionalProperties: false
    }
  }
}

// ── Settings ──────────────────────────────────────────────────────────
export interface Settings {
  // API endpoints
  apiBaseUrl: string
  apiKey: string
  mcpBaseUrl: string
  mcpApiKey: string

  // Model
  model: string

  // Generation — defaults change when enableThinking toggles
  temperature: number
  maxTokens: number
  topP: number
  topK: number
  frequencyPenalty: number
  presencePenalty: number

  // Agentic
  enableMcp: boolean
  enableThinking: boolean
  maxAgentIterations: number

  // Prompt
  systemPrompt: string

  // UI
  theme: 'light' | 'dark' | 'system'
  sidebarOpen: boolean
}

// ── Stream accumulation ───────────────────────────────────────────────
export interface ToolCallDraft {
  id: string
  name: string
  arguments: string
}

export interface StreamResult {
  content: string
  thinking: string
  toolCalls: ToolCall[]
  finishReason: 'stop' | 'tool_calls' | 'length' | 'error' | null
}
