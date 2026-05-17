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
import threading
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
    # Trigger model load (and potential download) in background
    asyncio.create_task(asyncio.to_thread(get_model))
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
_is_loading = False
_model_lock = threading.Lock()

def get_model():
    global _model, _is_loading
    if _model is None:
        with _model_lock:
            if _model is None and not _is_loading:
                _is_loading = True
                try:
                    from backend.model_loader import get_model as loader_get_model
                    _model = loader_get_model()
                finally:
                    _is_loading = False
    return _model

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        model = get_model()
        if model is None:
            return None
        vocab = get_vocab()
        _orchestrator = SystemOrchestrator(model, vocab)
    return _orchestrator

# --- Models ---
class ChatRequest(BaseModel):
    message: List[str]
    system_prompt: Optional[str] = "You are a highly intelligent Mixture of Experts (MoE) Large Language Model. You provide accurate, helpful, and concise answers."
    use_rag: bool = True
    use_web: bool = False
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 512
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
    if model is None:
        async def loading_gen():
            yield json.dumps({"word": "Model is still loading... ", "expert_id": 0, "expert_name": "System"}) + "\n"
        return StreamingResponse(loading_gen(), media_type="application/json")
    
    vocab = get_vocab()
    is_ext = getattr(model, "is_external", False)
    
    # Context retrieval
    context_str = ""
    if req.use_rag:
        retrieved = await rag.retrieve(req.message, k=5, use_web=req.use_web)
        if retrieved:
            context_str = "\n\nContext information:\n" + "\n".join([" ".join(c) for c in retrieved])
            
    user_query = " ".join(req.message)
    
    # Construct Chat Format with better context isolation
    full_persona = req.system_prompt
    if context_str:
        full_persona += f"\n\n### CONTEXT INFORMATION\nThe following information was retrieved to help you answer accurately:\n{context_str}\n\n### RESPONSE GUIDELINES\nUse the context above if relevant. If the context doesn't help, rely on your general knowledge. Maintain your persona as defined in the initial instructions."

    messages = [
        {"role": "system", "content": full_persona},
        {"role": "user", "content": user_query}
    ]
    
    # Prepare model name for display
    display_name = EXTERNAL_MODEL_PATH.split('/')[-1] if USE_EXTERNAL_MODEL else "System"

    async def generate():
        if is_ext:
            # True token-by-token streaming, non-blocking via asyncio.to_thread
            iterator = iter(model.adapter.stream_generate(
                messages, 
                max_new_tokens=req.max_tokens, 
                temperature=req.temperature
            ))
            final_text = ""
            while True:
                try:
                    token = await asyncio.to_thread(next, iterator)
                    final_text += token
                    yield json.dumps({
                        "word": token,
                        "expert_id": 0,
                        "expert_name": display_name
                    }) + "\n"
                except StopIteration:
                    break
            
            get_orchestrator().record_chat_interaction(req.message, final_text, 0, 0.95)
            return

        # Original Custom MoE Logic
        ids = torch.tensor([[get_word_id(w) for w in user_query.split()[-CONTEXT_LEN:]]]).long().to(DEVICE)
        model.eval()
        final_text = ""
        avg_confidence = 0.0
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
                
                # Confidence tracking
                conf = torch.max(probs).item()
                avg_confidence = (avg_confidence * i + conf) / (i + 1)
                
                word = vocab.get(str(predicted_id), "unknown")
                final_text += word + " "
                
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
                
        get_orchestrator().record_chat_interaction(req.message, final_text.strip(), expert_id.item(), avg_confidence)

    return StreamingResponse(generate(), media_type="application/json")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    persistence.increment_stat("total_interactions")
    model = get_model()
    if model is None:
        return ChatResponse(
            response="Model is still loading. Please wait...",
            expert_used=0,
            expert_name="System",
            confidence=1.0
        )
    vocab = get_vocab()
    orchestrator = get_orchestrator()
    is_ext = getattr(model, "is_external", False)
    
    # RAG retrieval & Context Integration
    context_str = ""
    retrieved_chunks = []
    if req.use_rag:
        retrieved_chunks = await rag.retrieve(req.message, k=5, use_web=req.use_web)
        if retrieved_chunks:
            context_str = "\n\nContext information:\n" + "\n".join([" ".join(c) for c in retrieved_chunks])
    
    user_query = " ".join(req.message)
    full_persona = req.system_prompt
    if context_str:
        full_persona += f"\n\n### CONTEXT INFORMATION\n{context_str}\n\n### GUIDELINES\nAnswer based on context if available. Maintain your persona."

    messages = [
        {"role": "system", "content": full_persona},
        {"role": "user", "content": user_query}
    ]

    # Prepare initial input
    if is_ext:
        final_response = await asyncio.to_thread(
            model.adapter.generate,
            messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature
        )
        avg_confidence = 0.95
        main_expert_id = 0
    else:
        gen_result = await asyncio.to_thread(
            generate_text,
            model, vocab, req.message, # Note: Legacy custom model doesn't support complex templates yet
            max_new_words=req.max_tokens, temperature=req.temperature, top_k=req.top_k, context_len=CONTEXT_LEN
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
    if orchestrator is None:
        return {
            "is_active": False,
            "current_stage": 0,
            "total_stages": 10,
            "stage_name": "Initializing Model...",
            "idle_time": 0,
            "idle_threshold": 60,
            "progress": []
        }
    status = orchestrator.dream.get_status()
    # Convert progress dict to sorted list for frontend
    if isinstance(status.get("progress"), dict):
        prog_dict = status["progress"]
        status["progress"] = [prog_dict.get(str(i), prog_dict.get(i, 0)) for i in range(10)]
    
    # Ensure total_stages is present for frontend progress bar
    status["total_stages"] = len(orchestrator.dream.curriculum) if hasattr(orchestrator.dream, 'curriculum') else 10
    return status

@app.post("/dream/start")
async def start_dream():
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    orchestrator.dream.start()
    return {"status": "started"}

@app.post("/dream/stop")
async def stop_dream():
    orchestrator = get_orchestrator()
    if orchestrator is None:
        return {"status": "already stopped"}
    orchestrator.dream.stop()
    return {"status": "stopped"}

@app.post("/dream/activity")
async def activity():
    orchestrator = get_orchestrator()
    if orchestrator:
        orchestrator.dream.record_activity()
    return {"status": "ok"}

@app.post("/dream/threshold")
async def set_threshold(body: dict):
    orchestrator = get_orchestrator()
    if orchestrator:
        orchestrator.dream.set_threshold(body.get("threshold", 60))
        return {"status": "updated"}
    return {"status": "skipped"}

@app.post("/build")
async def build_software(req: BuildRequest):
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Architect Engine is still loading")
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
    model_name = EXTERNAL_MODEL_PATH if USE_EXTERNAL_MODEL else "Local MoE"
    if model is None:
        return {
            "status": "loading",
            "expert_utilization": [0.0] * NUM_EXPERTS,
            "vocab_size": VOCAB_SIZE,
            "num_experts": NUM_EXPERTS,
            "device": str(DEVICE),
            "is_external": USE_EXTERNAL_MODEL,
            "model_name": model_name
        }
        
    is_ext = getattr(model, "is_external", False)
    
    # Dynamic values
    v_size = VOCAB_SIZE
    n_experts = NUM_EXPERTS
    
    if is_ext:
        v_size = model.adapter.tokenizer.vocab_size
        n_experts = 1 # Dense models are treated as 1 expert in the UI

    util = model.get_expert_utilization()
    expert_names = DOMAINS
    
    if isinstance(util, dict):
        # Handle dict-based expert names from external model
        expert_names = list(util.keys())
        processed_util = list(util.values())
        n_experts = len(processed_util)
    else:
        if torch.is_tensor(util):
            util = util.tolist()
        processed_util = util
        # Ensure we don't have more utils than names
        expert_names = DOMAINS[:len(processed_util)]

    # Check for restoration status
    restore_info = None
    restore_file = Path("data/restore_status.json")
    if restore_file.exists():
        try:
            with open(restore_file, "r") as f:
                restore_info = json.load(f)
        except:
            pass

    return {
        "expert_utilization": processed_util,
        "expert_names": expert_names,
        "vocab_size": v_size,
        "num_experts": n_experts,
        "device": str(DEVICE),
        "is_external": is_ext,
        "model_name": model_name,
        "status": "active",
        "restore_info": restore_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
