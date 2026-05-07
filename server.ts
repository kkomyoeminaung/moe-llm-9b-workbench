import express from "express";
import { createServer as createViteServer } from "vite";
import { createProxyMiddleware } from 'http-proxy-middleware';
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Start FastAPI backend
  console.log("Installing Python dependencies...");
  const installProcess = spawn("pip3", ["install", "-r", "requirements.txt"]);
  
  installProcess.on("error", (err) => {
    console.error("Failed to start pip3 installation:", err);
    // Try to start backend anyway
    startBackend();
  });

  installProcess.on("exit", (code) => {
    if (code === 0) {
      console.log("Python dependencies installed successfully.");
    } else {
      console.error(`Python dependency installation failed with code ${code}`);
    }
    startBackend();
  });

  function startBackend() {
    console.log("Starting FastAPI backend...");
    const logFile = path.join(process.cwd(), "backend.log");
    const logStream = fs.createWriteStream(logFile, { flags: 'w' }); // Overwrite for clarity

    const pythonProcess = spawn("python3", ["backend/app_unified.py"], {
      env: { ...process.env, PYTHONPATH: process.cwd() }
    });

    pythonProcess.stdout.pipe(logStream);
    pythonProcess.stderr.pipe(logStream);

    pythonProcess.on("error", (err) => {
      console.error("Failed to start FastAPI backend:", err);
      logStream.write(`Failed to start FastAPI backend: ${err.message}\n`);
    });

    pythonProcess.on("exit", (code) => {
      console.log(`FastAPI backend exited with code ${code}`);
      logStream.write(`FastAPI backend exited with code ${code}\n`);
    });
  }

  // Proxy /api requests to FastAPI backend
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8080',
      changeOrigin: true,
      pathRewrite: {
        '^/api': '', // remove /api prefix when forwarding
      },
      on: {
        error: (err, req, res) => {
          console.error('Proxy Error:', err);
          // @ts-ignore
          res.status(503).json({ error: 'Backend service unavailable' });
        }
      }
    })
  );

  // Catch-all to prevent /api from falling through to Vite
  app.use('/api', (req, res) => {
    res.status(404).json({ error: 'API route not found' });
  });

  // API Health Check
  app.get("/healthz", (req, res) => {
    res.json({ status: "ok" });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
