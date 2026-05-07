# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import torch
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "training"))
from config import *

from backend.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(title="Word MoE LLM API - Production Ready")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    message: List[str]
    use_rag: bool = True
    stream: bool = False
    temperature: float = 0.7
    top_k: int = 50

class ChatResponse(BaseModel):
    response: str
    expert_used: int
    expert_name: str
    confidence: float
    retrieved_chunks: Optional[List[List[str]]] = None

# Global variables (lazy loaded)
_model = None
_rag = None
_learner = None
_vocab = None
_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from backend.system_orchestrator import SystemOrchestrator
        model = get_model()
        vocab = get_vocab()
        _orchestrator = SystemOrchestrator(model, vocab)
    return _orchestrator

class BuildRequest(BaseModel):
    project_name: str
    requirements: str

@app.post("/build")
async def build_software(req: BuildRequest):
    orchestrator = get_orchestrator()
    result = orchestrator.build_software(req.project_name, req.requirements)
    return result

@app.get("/download/{filename}")
async def download_file(filename: str):
    from fastapi.responses import FileResponse
    file_path = Path("exports") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)

def get_model():
    global _model
    if _model is None:
        from backend.model_loader import get_model as loader_get_model
        _model = loader_get_model()
    return _model

def get_rag():
    global _rag
    if _rag is None:
        from rag_engine import RAGEngine
        _rag = RAGEngine()
    return _rag

def get_vocab():
    global _vocab
    if _vocab is None:
        _vocab = {i: f"word_{i}" for i in range(VOCAB_SIZE)}
        # Load actual vocab if exists
        vocab_path = Path("data/vocab.json")
        if vocab_path.exists():
            import json
            with open(vocab_path, "r") as f:
                _vocab = json.load(f)
    return _vocab

_word_to_idx = None
def get_word_to_idx():
    global _word_to_idx
    if _word_to_idx is None:
        vocab_path = Path("data/word_to_idx.json")
        if vocab_path.exists():
            import json
            with open(vocab_path, "r") as f:
                _word_to_idx = json.load(f)
        else:
            _word_to_idx = {}
    return _word_to_idx

def get_word_id(w: str) -> int:
    w2i = get_word_to_idx()
    if w in w2i:
        return w2i[w]
    import hashlib
    return int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16) % VOCAB_SIZE

# Streaming response generator
async def stream_response(message: List[str], temperature: float, top_k: int):
    model = get_model()
    vocab = get_vocab()
    word_ids = torch.tensor([[get_word_id(w) for w in message[-64:]]]).long()
    
    with torch.no_grad():
        outputs, expert_id = model(word_ids)
        probs = torch.softmax(outputs[0] / temperature, dim=-1)
        
        # Top-k sampling
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        top_k_probs = top_k_probs / top_k_probs.sum()
        
        # Sample from top-k
        sampled_idx = torch.multinomial(top_k_probs, 1).item()
        predicted_id = top_k_indices[sampled_idx].item()
        
        # Stream word by word (simulate)
        predicted_word = vocab.get(str(predicted_id), "unknown")
        
        # Stream as SSE
        for char in predicted_word:
            yield f"data: {json.dumps({'token': char, 'expert': expert_id.item(), 'done': False})}\n\n"
            await asyncio.sleep(0.05)
        
        yield f"data: {json.dumps({'token': '', 'expert': expert_id.item(), 'done': True})}\n\n"

@app.get("/")
async def root():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(open(frontend_path).read())
    return HTMLResponse("<h1>Word MoE LLM API</h1><p>Frontend not found. Please build frontend.</p>")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model = get_model()
    rag = get_rag()
    vocab = get_vocab()
    
    # Convert words to IDs
    word_ids = torch.tensor([[get_word_id(w) for w in req.message[-64:]]]).long()
    
    # RAG retrieval
    retrieved = []
    if req.use_rag:
        retrieved = rag.retrieve(req.message, k=5)
        # Optionally augment input with retrieved chunks
        if retrieved:
            context_words = retrieved[0][:10] if retrieved[0] else []
            if context_words:
                req.message = context_words + req.message
    
    # Generate response with temperature
    with torch.no_grad():
        outputs, expert_id = model(word_ids)
        probs = torch.softmax(outputs[0] / req.temperature, dim=-1)
        
        # Top-k sampling
        top_k_probs, top_k_indices = torch.topk(probs, req.top_k)
        top_k_probs = top_k_probs / top_k_probs.sum()
        sampled_idx = torch.multinomial(top_k_probs, 1).item()
        predicted_id = top_k_indices[sampled_idx].item()
        
        confidence = top_k_probs[sampled_idx].item()
        response_word = vocab.get(str(predicted_id), "unknown")
    
    expert_names = DOMAINS if 'DOMAINS' in dir() else ["chat", "engineering", "science", "medicine", "software_dev", "religion", "history", "economy", "politics", "literature"]
    
    return ChatResponse(
        response=response_word,
        expert_used=expert_id.item(),
        expert_name=expert_names[expert_id.item()] if expert_id.item() < len(expert_names) else "chat",
        confidence=confidence,
        retrieved_chunks=retrieved if retrieved else None
    )

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.stream:
        return await chat(req)
    return StreamingResponse(
        stream_response(req.message, req.temperature, req.top_k),
        media_type="text/event-stream"
    )

@app.get("/stats")
async def get_stats():
    model = get_model()
    return {
        "expert_utilization": model.get_expert_utilization().tolist() if hasattr(model, 'get_expert_utilization') else [0.1] * 10,
        "vocab_size": VOCAB_SIZE,
        "num_experts": NUM_EXPERTS,
        "device": "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
