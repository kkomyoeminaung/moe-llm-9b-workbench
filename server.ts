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

  function startBackend() {
    if (process.env.START_BACKEND === "true") {
      console.log("🚀 Starting PyTorch Backend Engine...");
      const backend = spawn("python3", ["backend/app_unified.py"]);
      backend.stdout.on("data", (data) => console.log(`[Backend] ${data}`));
      backend.stderr.on("data", (data) => console.error(`[Backend Error] ${data}`));
      backend.on("error", (err) => {
        console.error("❌ Failed to start python backend:", err.message);
      });
    } else {
      console.log("ℹ️ External Backend Mode: Proxying to port 8080. (Use START_BACKEND=true to run locally)");
    }
  }
  startBackend();

  // Middleware to mock responses if backend is not running in this container
  // Mock mode is disabled by default.
  const useMocks = false;
  
  if (useMocks) {
    console.log("⚠️ MOCK_AI is enabled. Intercepting /api routes with simulated MoE engine responses.");
    
    app.use(express.json());
    
    app.post('/api/chat', (req, res) => {
      res.json({
        response: "This is a simulated response indicating that the UI is functioning correctly. To use the real 7B MoE engine, please deploy using Docker or Google Colab as described in your Run Guide.",
        expert_used: 0,
        expert_name: "Mock AI Expert",
        confidence: 0.99,
        sources: ["System Note"]
      });
    });

    app.post('/api/chat/stream', (req, res) => {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      const text = "This is a streaming simulated response. The UI and streaming chunk ingestion is working perfectly. Please use `run_all.sh` in Colab or Kaggle to connect to the real PyTorch engine!";
      const words = text.split(' ');
      
      let i = 0;
      const interval = setInterval(() => {
        if (i >= words.length) {
          clearInterval(interval);
          res.end();
          return;
        }
        res.write(JSON.stringify({
          word: words[i] + ' ',
          expert_id: 0,
          expert_name: "Mock Streaming Expert"
        }) + '\\n');
        i++;
      }, 100);
    });

    app.post('/api/build', (req, res) => {
      setTimeout(() => {
        res.json({
          success: true,
          files: ["index.js", "utils.js", "package.json"],
          output: "Mock Build Successful. All logical constraints met. Generating artifact.",
          zip_url: "#",
          error: null
        });
      }, 3000);
    });

    app.get('/api/stats', (req, res) => {
      res.json({
        status: "ok",
        expert_utilization: [0.45, 0.35, 0.20],
        vocab_size: 32000,
        num_experts: 3,
        device: "cpu (mock)",
        is_external: true,
        model_name: "Simulated Model Status"
      });
    });

    app.get('/api/dream/status', (req, res) => {
      res.json({
        is_active: false,
        current_stage: 1,
        total_stages: 10,
        stage_name: "Idle (Mock)",
        idle_time: 0,
        idle_threshold: 60,
        progress: []
      });
    });

    app.post('/api/dream/activity', (req, res) => {
      res.json({ status: "ok" });
    });

    app.post('/api/ingest/upload', (req, res) => {
      setTimeout(() => {
        res.json({
          results: [{ success: true, message: "Mock File Uploaded" }]
        });
      }, 1000);
    });
  } else {
    console.log("✅ MOCK_AI is false. Production Mode: Proxying all /api traffic to PyTorch Engine (port 8080).");
  }

  const backendTarget = process.env.BACKEND_URL || 'http://127.0.0.1:8080';

  // Mock response fallback function
  function serveMockResponse(req: any, res: any) {
    if (req.url === '/chat/stream') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      });
      const text = "System is starting up... (PyTorch engine loading). Please copy your Session IP as the 'Tunnel Password' if requested. This usually takes 2-5 minutes in Colab/Kaggle environments.";
      const words = text.split(' ');
      let i = 0;
      const interval = setInterval(() => {
        if (i >= words.length) {
          clearInterval(interval);
          res.end();
          return;
        }
        res.write(JSON.stringify({
          word: words[i] + ' ',
          expert_id: 0,
          expert_name: "Mock AI (Preview Context)"
        }) + '\n');
        i++;
      }, 50);
      return;
    }

    if (req.url === '/stats') {
      return res.status(200).json({
        status: "mock",
        expert_utilization: [0.33, 0.33, 0.34],
        vocab_size: 32000,
        num_experts: 3,
        device: "cpu (mock)",
        is_external: true,
        model_name: "AI Studio Preview (No Backend)"
      });
    }

    if (req.url === '/dream/status') {
      return res.status(200).json({
        is_active: false,
        current_stage: 1,
        total_stages: 10,
        stage_name: "Idle (Mock)",
        idle_time: 0,
        idle_threshold: 60,
        progress: []
      });
    }

    if (req.url === '/chat') {
      return res.status(200).json({
        response: "The Python backend is not running in this environment. Please run the Kaggle/Colab notebook.",
        expert_used: 0,
        expert_name: "System",
        confidence: 1.0,
        sources: []
      });
    }

    return res.status(503).json({ error: 'Backend service unavailable', details: 'No python backend found' });
  }

  // Keep proxy for other routes just in case
  app.use(
    '/api',
    createProxyMiddleware({
      target: backendTarget,
      changeOrigin: true,
      pathRewrite: {
        '^/api': '', 
      },
      on: {
        proxyReq: (proxyReq, req, res) => {
           // Optionally do something before passing
        },
        error: (err, req, res) => {
          console.log(`⚠️ Backend unreachable (${err.message}), serving mock for ${req.url}`);
          serveMockResponse(req, res);
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
