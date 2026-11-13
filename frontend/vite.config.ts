import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run build` writes the app into the Python package, which `agent ui` serves.
// `npm run dev` serves it with hot reload and passes /api to a running `agent ui`.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../src/research_agent/web/static",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
