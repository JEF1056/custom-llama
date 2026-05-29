import { useState } from 'react'
import { X, RotateCcw, Eye, EyeOff } from 'lucide-react'
import { useSettingsStore, DEFAULT_SYSTEM_PROMPT } from '../store/settingsStore'
import type { Settings } from '../types'

interface Props {
  open: boolean
  onClose: () => void
}

export function SettingsModal({ open, onClose }: Props) {
  const { settings, updateSettings, resetSettings, toggleThinking } = useSettingsStore()
  const [showApiKey, setShowApiKey] = useState(false)
  const [showMcpKey, setShowMcpKey] = useState(false)

  if (!open) return null

  function patch<K extends keyof Settings>(key: K, value: Settings[K]) {
    updateSettings({ [key]: value })
  }

  function handleThinkingToggle(enabled: boolean) {
    toggleThinking(enabled)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-[var(--bg-surface)] border border-[var(--border)] rounded-3xl shadow-[0_20px_60px_rgba(20,20,20,0.24)] w-full max-w-lg max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] shrink-0">
          <h2 className="text-base font-semibold text-[var(--text-primary)]">Settings</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { resetSettings(); }}
              title="Reset to defaults"
              className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] px-2.5 py-1.5 rounded-lg hover:bg-[var(--bg-elevated)] transition-colors"
            >
              <RotateCcw size={13} />
              Reset
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="overflow-y-auto flex-1 px-6 py-5 space-y-6">
          {/* API */}
          <Section title="API">
            <Field label="Base URL">
              <Input
                value={settings.apiBaseUrl}
                onChange={(v) => patch('apiBaseUrl', v)}
                placeholder="https://api.jessfan.com/v1"
              />
            </Field>
            <Field label="API Key">
              <div className="relative">
                <Input
                  type={showApiKey ? 'text' : 'password'}
                  value={settings.apiKey}
                  onChange={(v) => patch('apiKey', v)}
                  placeholder="sk-…"
                />
                <button
                  onClick={() => setShowApiKey((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                >
                  {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              <p className="text-xs text-[var(--text-secondary)] mt-1">Stored in localStorage only.</p>
            </Field>
            <Field label="Model">
              <Input
                value={settings.model}
                onChange={(v) => patch('model', v)}
                placeholder="Qwen3-8B (leave empty for server default)"
              />
            </Field>
          </Section>

          {/* MCP */}
          <Section title="MCP Server">
            <Field label="">
              <Toggle
                label="Enable MCP tools"
                checked={settings.enableMcp}
                onChange={(v) => patch('enableMcp', v)}
              />
            </Field>
            <Field label="MCP Base URL">
              <Input
                value={settings.mcpBaseUrl}
                onChange={(v) => patch('mcpBaseUrl', v)}
                placeholder="https://mcp.jessfan.com"
                disabled={!settings.enableMcp}
              />
            </Field>
            <Field label="MCP API Key">
              <div className="relative">
                <Input
                  type={showMcpKey ? 'text' : 'password'}
                  value={settings.mcpApiKey}
                  onChange={(v) => patch('mcpApiKey', v)}
                  placeholder="Bearer token"
                  disabled={!settings.enableMcp}
                />
                <button
                  onClick={() => setShowMcpKey((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                >
                  {showMcpKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </Field>
          </Section>

          {/* Generation */}
          <Section title="Generation">
            <Field label="">
              <Toggle
                label="Thinking mode (Qwen3)"
                checked={settings.enableThinking}
                onChange={handleThinkingToggle}
              />
            </Field>
            <Field label={`Temperature: ${settings.temperature}`}>
              <RangeInput
                min={0.1}
                max={2}
                step={0.05}
                value={settings.temperature}
                onChange={(v) => patch('temperature', v)}
              />
            </Field>
            <Field label={`Max tokens: ${settings.maxTokens}`}>
              <RangeInput
                min={256}
                max={32768}
                step={256}
                value={settings.maxTokens}
                onChange={(v) => patch('maxTokens', v)}
              />
            </Field>
            <Field label={`Top-P: ${settings.topP}`}>
              <RangeInput
                min={0.1}
                max={1}
                step={0.05}
                value={settings.topP}
                onChange={(v) => patch('topP', v)}
              />
            </Field>
            <Field label={`Top-K: ${settings.topK}`}>
              <RangeInput
                min={1}
                max={100}
                step={1}
                value={settings.topK}
                onChange={(v) => patch('topK', v)}
              />
            </Field>
            <Field label={`Frequency penalty: ${settings.frequencyPenalty}`}>
              <RangeInput
                min={0}
                max={2}
                step={0.05}
                value={settings.frequencyPenalty}
                onChange={(v) => patch('frequencyPenalty', v)}
              />
            </Field>
            <Field label={`Presence penalty: ${settings.presencePenalty}`}>
              <RangeInput
                min={0}
                max={2}
                step={0.05}
                value={settings.presencePenalty}
                onChange={(v) => patch('presencePenalty', v)}
              />
            </Field>
            <Field label={`Max agent iterations: ${settings.maxAgentIterations}`}>
              <RangeInput
                min={1}
                max={20}
                step={1}
                value={settings.maxAgentIterations}
                onChange={(v) => patch('maxAgentIterations', v)}
              />
            </Field>
          </Section>

          {/* System prompt */}
          <Section title="System Prompt">
            <Field label="">
              <textarea
                value={settings.systemPrompt}
                onChange={(e) => patch('systemPrompt', e.target.value)}
                rows={10}
                className="w-full px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--bg-base)] text-sm text-[var(--text-primary)] font-mono resize-y focus:outline-none focus:ring-2 focus:ring-black/10 dark:focus:ring-white/10"
              />
              <button
                onClick={() => patch('systemPrompt', DEFAULT_SYSTEM_PROMPT)}
                className="mt-1 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              >
                ↩ Restore default
              </button>
            </Field>
          </Section>
        </div>
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-secondary)]">{title}</h3>
      <div className="space-y-3">{children}</div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      {label && <label className="text-xs text-[var(--text-secondary)]">{label}</label>}
      {children}
    </div>
  )
}

function Input({
  value,
  onChange,
  placeholder,
  type = 'text',
  disabled,
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
  type?: string
  disabled?: boolean
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      className="w-full px-4 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--bg-base)] text-sm text-[var(--text-primary)] placeholder:text-[var(--placeholder)] focus:outline-none focus:ring-2 focus:ring-black/10 dark:focus:ring-white/10 disabled:opacity-50"
    />
  )
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="flex items-center gap-3 cursor-pointer">
      <div
        className={`relative w-9 h-5 rounded-full transition-colors ${checked ? 'bg-[var(--text-primary)]' : 'bg-[var(--border)]'}`}
        onClick={() => onChange(!checked)}
      >
        <div
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-4' : ''}`}
        />
      </div>
      <span className="text-sm text-[var(--text-primary)]">{label}</span>
    </label>
  )
}

function RangeInput({
  min,
  max,
  step,
  value,
  onChange,
}: {
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
}) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full accent-[var(--text-primary)]"
    />
  )
}
