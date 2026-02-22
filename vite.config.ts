import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // IMPORTANT for GitHub Pages project repo:
  base: "/MedDecode-AI/",
});
