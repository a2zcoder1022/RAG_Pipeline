import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const projectRoot = fileURLToPath(new URL('../', import.meta.url))

export default defineConfig({
  plugins: [react()],
  cacheDir: join(tmpdir(), 'rag-pipeline-vite-cache'),
  server: {
    fs: {
      allow: [projectRoot],
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
