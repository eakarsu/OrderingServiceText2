import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// React plugin already covers .tsx/.jsx; for the spec-mandated *.js JSX files
// under src/, register a tiny plugin that hands them to esbuild as JSX.
const jsAsJsx = {
  name: 'js-as-jsx',
  async transform(code: string, id: string) {
    if (!id.includes('/src/')) return null
    if (!id.endsWith('.js')) return null
    const { transformWithEsbuild } = await import('vite')
    return transformWithEsbuild(code, id, { loader: 'jsx', jsx: 'automatic' })
  },
}

export default defineConfig({
  plugins: [jsAsJsx, react(), tailwindcss()],
  optimizeDeps: {
    esbuildOptions: {
      loader: { '.js': 'jsx' },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
