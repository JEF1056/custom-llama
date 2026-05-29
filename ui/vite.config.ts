import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Dev-only: proxy /v1 → local SGLang and /mcp-dev → local MCP server
      '/v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/mcp-dev': {
        target: 'http://localhost:3100',
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
