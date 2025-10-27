import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  root: '.',  // Explicitly set root directory
  publicDir: 'public',  // Directory for static assets
  server: {
    host: "::",
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
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
