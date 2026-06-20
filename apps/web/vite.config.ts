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
      // NOTE: do NOT proxy "/docs" — it collides with the SPA's own /docs route
      // (DocsPage). Proxying it makes a hard load / reload of /docs serve the
      // API's Swagger UI instead of the app. The API's Swagger UI is reachable
      // directly at the gateway origin (:8000/docs) when needed.
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    css: false,
  },
});
