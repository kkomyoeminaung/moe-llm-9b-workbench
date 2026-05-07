# backend/app_unified.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import torch
import asyncio
import json
import sys
import os
from pathlib import Path

# Add training folder to path
sys.path.append(str(Path(__file__).parent.parent / "training"))
from config import *

from backend.logger import setup_logger
from backend.persistence_auto import get_persistence
from backend.integrated_rag import IntegratedRAG
from backend.continuous_learner_persistent import PersistentContinuousLearner
from backend.knowledge_ingestion import KnowledgeIngestion
from backend.dream_mode import DreamMode
from backend.self_learning import SelfLearningSystem
from backend.system_orchestrator import SystemOrchestrator
from backend.utils import get_vocab, get_word_id, generate_text

logger = setup_logger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MoE LLM Unified Backend")
    yield
    # Shutdown
    logger.info("Shutting down MoE LLM Unified Backend")
    rag.save()
    persistence.save_all()
    await rag.close()

app = FastAPI(title="MoE LLM - Unified Intelligence Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# --- Global Components ---
persistence = get_persistence()
rag = IntegratedRAG()
learner = PersistentContinuousLearner()

# Global variables (lazy loaded)
_model = None
_orchestrator = None
_vocab = None
_word_to_idx = None

def get_model():
    global _model
    if _model is None:
        from backend.model_loader import get_model as loader_get_model
        _model = loader_get_model()
    return _model

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        model = get_model()
        vocab = get_vocab()
        _orchestrator = SystemOrchestrator(model, vocab)
    return _orchestrator

# --- Models ---
class ChatRequest(BaseModel):
    message: List[str]
    use_rag: bool = True
    use_web: bool = False
    stream: bool = False
    temperature: float = 0.7
    top_k: int = 50

class ChatResponse(BaseModel):
    response: str
    expert_used: int
    expert_name: str
    confidence: float
    sources: Optional[List[str]] = None

class BuildRequest(BaseModel):
    project_name: str
    requirements: str

# --- Endpoints ---

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    model = get_model()
    vocab = get_vocab()
    
    # Context retrieval
    context_prefix = []
    if req.use_rag:
        retrieved = await rag.retrieve(req.message, k=5, use_web=req.use_web)
        for chunk in retrieved:
            context_prefix.extend(chunk[:5])
            
    raw_input = (context_prefix + req.message)[-CONTEXT_LEN:]
    current_ids = torch.tensor([[get_word_id(w) for w in raw_input]]).long().to(DEVICE)
    
    async def generate():
        ids = current_ids.clone()
        model.eval()
        with torch.no_grad():
            for i in range(50): # Max stream length
                outputs, expert_id = model(ids)
                logits = outputs[0, -1, :] if outputs.dim() == 3 else outputs[0]
                probs = torch.softmax(logits / req.temperature, dim=-1)
                
                # Sample
                top_p, top_i = torch.topk(probs, min(req.top_k, VOCAB_SIZE))
                top_p /= top_p.sum()
                sampled = torch.multinomial(top_p, 1).item()
                predicted_id = top_i[sampled].item()
                
                word = vocab.get(str(predicted_id), "unknown")
                
                # Yield JSON chunk
                yield json.dumps({
                    "word": word,
                    "expert_id": expert_id.item(),
                    "expert_name": DOMAINS[expert_id.item()] if expert_id.item() < len(DOMAINS) else "general"
                }) + "\n"
                
                if word in [".", "!", "?", "<eos>"] and i > 5:
                    break
                    
                new_id = torch.tensor([[predicted_id]]).long().to(DEVICE)
                ids = torch.cat([ids, new_id], dim=1)[:, -CONTEXT_LEN:]
                await asyncio.sleep(0.01)

    return StreamingResponse(generate(), media_type="application/json")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    persistence.increment_stat("total_interactions")
    model = get_model()
    vocab = get_vocab()
    orchestrator = get_orchestrator()
    
    # RAG retrieval & Context Integration
    context_prefix = []
    retrieved_chunks = []
    if req.use_rag:
        retrieved_chunks = await rag.retrieve(req.message, k=5, use_web=req.use_web)
        if retrieved_chunks:
            # Extract first keyword from each chunk to augment context
            for chunk in retrieved_chunks:
                context_prefix.extend(chunk[:5])
    
    # Prepare initial input
    gen_result = generate_text(
        model, vocab, context_prefix + req.message,
        max_new_words=30, temperature=req.temperature, top_k=req.top_k, context_len=CONTEXT_LEN
    )
    
    final_response = gen_result["text"]
    avg_confidence = gen_result["avg_confidence"]
    main_expert_id = gen_result["main_expert_id"]
    
    # Record for learning (using the full generated text as feedback for next time)
    orchestrator.record_chat_interaction(req.message, final_response, main_expert_id, avg_confidence)
    
    expert_name = DOMAINS[main_expert_id] if main_expert_id < len(DOMAINS) else "general"
    
    return ChatResponse(
        response=final_response,
        expert_used=main_expert_id,
        expert_name=expert_name,
        confidence=avg_confidence,
        sources=[f"Source {i+1}" for i in range(len(retrieved_chunks))] if retrieved_chunks else None
    )

@app.post("/ingest/upload")
async def upload_files(files: List[UploadFile] = File(...), domain: Optional[int] = Form(None)):
    orchestrator = get_orchestrator()
    results = []
    for file in files:
        content = await file.read()
        result = orchestrator.ingestion.process_upload(content, file.filename, domain=domain)
        results.append(result)
        rag.save()
    return {"results": results}

@app.get("/dream/status")
async def get_dream_status():
    orchestrator = get_orchestrator()
    status = orchestrator.dream.get_status()
    # Ensure total_stages is present for frontend progress bar
    status["total_stages"] = len(orchestrator.dream.curriculum) if hasattr(orchestrator.dream, 'curriculum') else 10
    return status

@app.post("/dream/start")
async def start_dream():
    orchestrator = get_orchestrator()
    orchestrator.dream.start()
    return {"status": "started"}

@app.post("/dream/stop")
async def stop_dream():
    orchestrator = get_orchestrator()
    orchestrator.dream.stop()
    return {"status": "stopped"}

@app.post("/dream/activity")
async def activity():
    orchestrator = get_orchestrator()
    orchestrator.dream.record_activity()
    return {"status": "ok"}

@app.post("/dream/threshold")
async def set_threshold(body: dict):
    orchestrator = get_orchestrator()
    orchestrator.dream.set_threshold(body.get("threshold", 60))
    return {"status": "updated"}

@app.post("/build")
async def build_software(req: BuildRequest):
    orchestrator = get_orchestrator()
    result = orchestrator.build_software(req.project_name, req.requirements)
    return result

@app.get("/download/{filename}")
async def download_file(filename: str):
    # Sanitize: only allow basename, prevent path traversal
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = Path("exports").resolve() / safe_filename
    exports_dir = Path("exports").resolve()

    # Ensure file is inside exports directory
    if not str(file_path).startswith(str(exports_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(path=file_path, filename=safe_filename)

@app.get("/stats")
async def get_stats():
    model = get_model()
    util = model.get_expert_utilization() if hasattr(model, 'get_expert_utilization') else [0.1] * 10
    if torch.is_tensor(util):
        util = util.tolist()
    return {
        "expert_utilization": util,
        "vocab_size": VOCAB_SIZE,
        "num_experts": NUM_EXPERTS,
        "device": str(DEVICE)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
