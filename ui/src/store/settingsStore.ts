import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Settings } from '../types'

const IS_DEV = import.meta.env.DEV

const DEFAULT_SYSTEM_PROMPT = `You are a helpful, knowledgeable AI assistant with access to tools and an MCP file server.

## Core behavior
- Resolve the user's request completely before yielding control back.
- Work in a Thought → Action → Observation cycle: reason before every tool call, interpret every result before the next step.
- If required information is missing and retrievable via a tool, use the tool — do not guess.
- If you must proceed without full context, state your assumptions explicitly and choose reversible actions.

## Tool use
- Use tools when you need real-time information, computation, file access, or web search.
- Prefer the most specific tool for the task. If multiple tools could apply, the more targeted one is better.
- After a tool returns, reflect: does this result fully address the goal, or are further steps needed?
- If a tool returns an error, analyze the cause and try a different approach — do not repeat the same failing call.
- Validate tool results before treating them as ground truth (tools can return success even on partial failure).

## File handling
- When a user attaches a file, check the message for its MCP filename and the recommended tool.
- For Excel files: use xlsx_read(filename="…") to read full data beyond the preview.
- For PowerPoint files: use pptx_read(filename="…") to extract text, pptx_slide_image(filename="…", slide_index=N) for images.
- For other files: use file_read(filename="…") to access content.
- All MCP filenames are bare names (no paths) — never include / or \\ in the filename argument.

## Self-check (after each major step)
- Does the output fully address what was asked?
- Are factual claims grounded in retrieved data or tool results?
- Did any tool return an ambiguous or partial result that should be flagged?

## Final answer
- Always deliver a complete final answer once tool use is done.
- Format with markdown when helpful. Include citations when results come from tool calls.
- Do not end your turn until the user's request is completely resolved.`

const DEFAULT_SETTINGS: Settings = {
  apiBaseUrl: IS_DEV ? '/v1' : 'https://api.jessfan.com/v1',
  apiKey: '',
  mcpBaseUrl: IS_DEV ? 'http://localhost:3100' : 'https://mcp.jessfan.com',
  mcpApiKey: '',
  model: '',
  temperature: 0.6,
  maxTokens: 4096,
  topP: 0.95,
  topK: 20,
  frequencyPenalty: 0,
  presencePenalty: 0,
  enableMcp: false,
  enableThinking: true,
  maxAgentIterations: 10,
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  theme: 'system',
  sidebarOpen: true,
}

interface SettingsStore {
  settings: Settings
  updateSettings: (patch: Partial<Settings>) => void
  resetSettings: () => void
  /** Called when enableThinking changes — adjusts temp/topP to Qwen3 official values
   *  unless the user has already manually overridden them from the defaults. */
  toggleThinking: (enabled: boolean) => void
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set, get) => ({
      settings: DEFAULT_SETTINGS,

      updateSettings: (patch) =>
        set((state) => ({ settings: { ...state.settings, ...patch } })),

      resetSettings: () => set({ settings: DEFAULT_SETTINGS }),

      toggleThinking: (enabled) => {
        const { settings } = get()
        // Only auto-adjust temp/topP if they're still at the OTHER mode's default
        const thinkingDefaults = { temperature: 0.6, topP: 0.95 }
        const nonThinkingDefaults = { temperature: 0.7, topP: 0.8 }
        const currentDefaults = enabled ? nonThinkingDefaults : thinkingDefaults
        const newDefaults = enabled ? thinkingDefaults : nonThinkingDefaults

        const patch: Partial<Settings> = { enableThinking: enabled }
        if (settings.temperature === currentDefaults.temperature) patch.temperature = newDefaults.temperature
        if (settings.topP === currentDefaults.topP) patch.topP = newDefaults.topP
        set((state) => ({ settings: { ...state.settings, ...patch } }))
      },
    }),
    {
      name: 'chat-settings',
      // Don't persist theme so system preference is always respected on first load
      partialize: (state) => state.settings,
    },
  ),
)

export { DEFAULT_SYSTEM_PROMPT }
