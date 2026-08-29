import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Сборка кладётся в studio/static/ и коммитится в репозиторий:
// пользователь без Node должен уметь запустить студию (решение D-11).
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../static",
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: {
    proxy: { "/api": "http://localhost:8080" },
  },
});
