import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The API gateway runs on :8000 in dev. Vite proxies the API surface so the
// browser talks same-origin and SSE streams flow through untouched.
const API_TARGET = process.env.ASTEL_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: API_TARGET, changeOrigin: true },
      "/healthz": { target: API_TARGET, changeOrigin: true },
      "/openapi.json": { target: API_TARGET, changeOrigin: true },
      "/docs": { target: API_TARGET, changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    css: false,
  },
});
