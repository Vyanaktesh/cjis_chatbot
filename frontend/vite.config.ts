import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Vite doesn't read process.env.PORT itself -- picked up explicitly so
    // an external harness assigning a port via that env var (see
    // .claude/launch.json's autoPort) actually takes effect instead of
    // always falling back to Vite's hardcoded default.
    port: Number(process.env.PORT) || 5173,
  },
})
