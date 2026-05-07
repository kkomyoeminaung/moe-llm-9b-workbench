# backend/app_complete.py
"""Complete FastAPI app with persistence and web search RAG"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import sys
from pathlib import Path

from backend.persistence_auto import get_persistence
from backend.integrated_rag import IntegratedRAG
from backend.continuous_learner_persistent import PersistentContinuousLearner
from backend.knowledge_ingestion import KnowledgeIngestion
from backend.dream_mode import DreamMode
from training.model_unified import SparseMoE_Unified

# Load configuration
sys.path.append(str(Path(__file__).parent.parent / "training"))
from config import DEVICE, VOCAB_SIZE

app = FastAPI(title="MoE LLM - Complete Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all components
persistence = get_persistence()
rag = IntegratedRAG()
learner = PersistentContinuousLearner()

# Model
model = SparseMoE_Unified(vocab_size=VOCAB_SIZE).to(DEVICE)
model.eval()

# Ingestion & Dream Mode
ingestion = KnowledgeIngestion(rag, learner, model)
dream_mode = DreamMode(model, learner, rag)

# Request models
class ChatRequest(BaseModel):
    message: List[str]
    use_rag: bool = True
    use_web: bool = False
    
@app.on_event("shutdown")
async def shutdown():
    rag.save()
    persistence.save_all()
    await rag.close()

@app.post("/chat")
async def chat(req: ChatRequest):
    persistence.increment_stat("total_interactions")
    retrieved_chunks = await rag.retrieve(req.message, k=3, use_web=req.use_web)
    
    # Simple response generation logic using retrieved chunks
    if retrieved_chunks:
        context = " ".join([" ".join(c) for c in retrieved_chunks[:2]])
        response = f"Based on retrieved context ({context[:50]}...): I understand your query about '{' '.join(req.message)}'"
    else:
        response = f"I understand your query: '{' '.join(req.message)}'. I don't have specific context retrieved."
    
    learner.store_episode(req.message, response, 0, 0.8)
    return {
        "response": response, 
        "expert_used": 0, 
        "sources": [f"Source {i+1}" for i in range(len(retrieved_chunks))] if retrieved_chunks else []
    }

@app.post("/ingest/upload")
async def upload_files(files: List[UploadFile] = File(...), domain: Optional[int] = Form(None)):
    results = []
    for file in files:
        content = await file.read()
        result = ingestion.process_upload(content, file.filename, domain=domain)
        results.append(result)
        rag.save()
    return {"results": results}

@app.get("/dream/status")
async def get_dream_status():
    return dream_mode.get_status()

@app.post("/dream/start")
async def start_dream():
    dream_mode.start()
    return {"status": "started"}

@app.post("/dream/stop")
async def stop_dream():
    dream_mode.stop()
    return {"status": "stopped"}

@app.post("/dream/activity")
async def activity():
    dream_mode.record_activity()
    return {"status": "ok"}

@app.post("/dream/threshold")
async def set_threshold(body: dict):
    dream_mode.set_threshold(body.get("threshold", 60))
    return {"status": "updated"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/ready")
async def ready():
    # Check if model is loaded and RAG is accessible
    is_ready = model is not None and rag is not None
    return {"ready": is_ready}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
