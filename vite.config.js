import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  base: '/',
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
      },
      '/config': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ingest': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/collections': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/dashboard/stats': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/dashboard/points': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/chat/stream': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    }
  }
});