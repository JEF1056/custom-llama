import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    proxy: {
      // Dev-only: proxy /v1 → vLLM and /mcp-dev → MCP server.
      // Override targets via env vars when running inside Docker.
      '/v1': {
        target: process.env.PROXY_API_TARGET ?? 'http://localhost:8080',
        changeOrigin: true,
      },
      '/mcp-dev': {
        target: process.env.PROXY_MCP_TARGET ?? 'http://localhost:3100',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/mcp-dev/, ''),
      },
    },
  },
  optimizeDeps: {
    // pdfjs and xlsx are large; pre-bundle them for faster dev HMR
    include: ['pdfjs-dist'],
    exclude: ['xlsx'],
  },
})
