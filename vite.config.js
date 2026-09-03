import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  base: './',
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/api': {
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
      },
      // 1. Vector Search (The one causing your 405/404)

      // 2. LLM Configuration & Model Fetching
      '/config': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // 3. RAG Chat & Streaming
      '/chat': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // 4. Data Ingestion (File uploads, Web URLs, HF Datasets)
      '/ingest': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // 5. Collection Management (Switching/Creating)
      '/collections': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // 6. Dashboard Stats & System Health
      '/dashboard': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
});