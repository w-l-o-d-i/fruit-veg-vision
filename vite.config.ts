import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    wasm(),
    topLevelAwait(),
    react(), 
    mode === "development" && componentTagger()
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    exclude: ["@tensorflow/tfjs-tflite"],
  },
  build: {
    target: "esnext",
    // Ensure large files like the model are handled correctly
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        // Keep model files in their original location
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith('.tflite')) {
            return 'models/[name][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        },
      },
    },
  },
}));
