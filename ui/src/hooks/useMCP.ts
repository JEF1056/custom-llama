/**
 * MCP tool discovery and execution hook.
 * Manages the client lifecycle, fetches available tools, and exposes callTool().
 */
import { useState, useEffect, useCallback } from 'react'
import type { MCPTool, OpenAITool } from '../types'
import { getMcpClient, toOpenAITool } from '../lib/mcp'
import { useSettingsStore } from '../store/settingsStore'

export interface UseMCPResult {
  tools: OpenAITool[]
  knownToolNames: Set<string>
  callTool: (name: string, args: Record<string, unknown>) => Promise<string>
  isLoading: boolean
  error: string | null
  reload: () => void
}

export function useMCP(): UseMCPResult {
  const { settings } = useSettingsStore()
  const [tools, setTools] = useState<OpenAITool[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  const { mcpBaseUrl, mcpApiKey, enableMcp } = settings

  useEffect(() => {
    if (!enableMcp || !mcpBaseUrl) {
      setTools([])
      setError(null)
      return
    }

    let cancelled = false
    setIsLoading(true)
    setError(null)

    const client = getMcpClient(mcpBaseUrl, mcpApiKey)

    async function load() {
      try {
        // Initialize then list tools
        await client.initialize()
        const mcpTools: MCPTool[] = await client.listTools()
        if (!cancelled) {
          setTools(mcpTools.map(toOpenAITool))
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
          setTools([])
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [mcpBaseUrl, mcpApiKey, enableMcp, tick])

  const callTool = useCallback(
    async (name: string, args: Record<string, unknown>): Promise<string> => {
      const client = getMcpClient(mcpBaseUrl, mcpApiKey)
      return client.callTool(name, args)
    },
    [mcpBaseUrl, mcpApiKey],
  )

  const reload = useCallback(() => {
    setTick((t) => t + 1)
  }, [])

  const knownToolNames = new Set(tools.map((t) => t.function.name))

  return { tools, knownToolNames, callTool, isLoading, error, reload }
}
