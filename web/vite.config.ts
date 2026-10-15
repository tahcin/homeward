import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built site goes to dist/, which wrangler serves as static assets in front of worker/index.ts.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false },
  // In development the Worker runs on wrangler's port; drop Origin so its same-site check sees a direct call.
  server: {
    proxy: {
      "/api": { target: "http://localhost:8787", configure: (proxy) => proxy.on("proxyReq", (req) => req.removeHeader("origin")) },
    },
  },
});
