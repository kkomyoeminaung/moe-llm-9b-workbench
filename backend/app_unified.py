# backend/app_unified.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator, Union
import torch
import asyncio
import psutil
import json
import sys
import os
import threading
import logging
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
from backend.utils import get_vocab, get_word_id, generate_text, tokenize

logger = setup_logger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MoE LLM Unified Backend")
    # Trigger model and component load in background
    async def init_components():
        try:
            get_rag()
            get_model()
            get_learner()
        except Exception as e:
            logger.error(f"Error during component initialization: {e}")
            
    asyncio.create_task(init_components())
    yield
    # Shutdown
    logger.info("Shutting down MoE LLM Unified Backend")
    if _rag: _rag.save()
    if _persistence: _persistence.save_all()
    if _rag: await _rag.close()

app = FastAPI(title="MoE LLM - Unified Intelligence Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Global components (lazy loaded)
_persistence = None
_rag = None
_learner = None

def get_persistence_cached():
    global _persistence
    if _persistence is None:
        _persistence = get_persistence()
    return _persistence

def get_rag():
    global _rag
    if _rag is None:
        _rag = IntegratedRAG()
    return _rag

def get_learner():
    global _learner
    if _learner is None:
        _learner = PersistentContinuousLearner()
    return _learner

# Global variables (lazy loaded)
_model = None
_orchestrator = None
_vocab = None
_word_to_idx = None
_is_loading = False
_model_lock = threading.Lock()

def get_model():
    global _model, _is_loading
    if _model is not None:
        return _model
    
    # If already loading, return None immediately to avoid blocking the main thread/worker
    if _is_loading:
        return None
        
    with _model_lock:
        # Check again inside lock to avoid race conditions
        if _model is not None:
            return _model
        if _is_loading:
            return None
            
        _is_loading = True
        try:
            from backend.model_loader import get_model as loader_get_model
            _model = loader_get_model()
        except Exception as e:
            logger.error(f"Critical error loading model: {e}")
            _model = None
        finally:
            _is_loading = False
            
    return _model

def get_orchestrator():
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator
        
    model = get_model()
    if model is None:
        return None
        
    with _model_lock: # Reuse lock for orchestrator initialization
        if _orchestrator is None:
            vocab = get_vocab()
            _orchestrator = SystemOrchestrator(model, vocab, get_rag(), get_learner())
    return _orchestrator

# --- Models ---
class ChatRequest(BaseModel):
    message: Union[str, List[str]]
    system_prompt: Optional[str] = "You are a highly intelligent Mixture of Experts (MoE) Large Language Model. You provide accurate, helpful, detailed, and eloquent answers."
    use_rag: bool = True
    use_web: bool = False
    stream: bool = False
    temperature: float = 0.1
    max_tokens: int = 1024
    top_k: int = 40

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
            yield f"data: {json.dumps({'word': 'Model is still loading... ', 'expert_id': 0, 'expert_name': 'System'})}\n\n"
        return StreamingResponse(loading_gen(), media_type="text/event-stream")
    
    # Ensure message is a list of words
    if isinstance(req.message, str):
        req.message = tokenize(req.message)
    elif isinstance(req.message, list):
        # Even if it's a list, if we have one big string in it, tokenize it
        if len(req.message) == 1 and " " in req.message[0]:
             req.message = tokenize(req.message[0])
    elif not req.message:
        req.message = ["hello"] # Default fallback
    
    vocab = get_vocab()
    is_ext = getattr(model, "is_external", False)
    
    # Prepare model name for display
    display_name = EXTERNAL_MODEL_PATH.split('/')[-1] if USE_EXTERNAL_MODEL else "System"

    async def generate():
        # 1. Immediate Initial Connection Feedback
        yield f"data: {json.dumps({'word': '', 'expert_id': 0, 'expert_name': 'MoE System', 'confidence': 0.0})}\n\n"
        
        # 2. Context retrieval INSIDE generator to avoid blocking the initial stream
        context_str = ""
        if req.use_rag:
            try:
                yield f"data: {json.dumps({'word': '🔍 [Diagnostic] Query reached RAG Engine. Searching knowledge base...\n', 'expert_id': 0, 'expert_name': 'RAG Engine'})}\n\n"
                retrieved = await get_rag().retrieve(req.message, k=5, use_web=req.use_web)
                if retrieved:
                    context_str = "\n\nContext information:\n" + "\n".join([" ".join(c) for c in retrieved])
                    yield f"data: {json.dumps({'word': '✅ [Diagnostic] Context integration complete. Routing to Expert...\n\n', 'expert_id': 0, 'expert_name': 'MoE System'})}\n\n"
                else:
                    yield f"data: {json.dumps({'word': 'ℹ️ [Diagnostic] No direct context found. Using general knowledge.\n\n', 'expert_id': 0, 'expert_name': 'MoE System'})}\n\n"
            except Exception as e:
                logger.error(f"RAG Retrieval Error: {e}")
                yield f"data: {json.dumps({'word': '⚠️ [Diagnostic] RAG Engine encountered an error, falling back to pure LLM.\n\n', 'expert_id': 0, 'expert_name': 'MoE System'})}\n\n"

        # Construct Final Prompt
        user_query = " ".join(req.message)
        full_persona = req.system_prompt
        if context_str:
            full_persona += f"\n\n### CONTEXT INFORMATION\nThe following information was retrieved to help you answer accurately:\n{context_str}\n\n### RESPONSE GUIDELINES\nUse the context above if relevant. Maintain your persona."

        messages = [
            {"role": "system", "content": full_persona},
            {"role": "user", "content": user_query}
        ]

        if is_ext:
            # Send expert identification for 7B model
            yield f"data: {json.dumps({'word': '🧠 MoE Architect thinking...\n\n', 'expert_id': 0, 'expert_name': display_name})}\n\n"
            
            # Optimized for 7B Qwen stability
            actual_temp = max(0.01, min(req.temperature, 0.9))
            
            try:
                iterator = iter(model.adapter.stream_generate(
                    messages, 
                    max_new_tokens=req.max_tokens, 
                    temperature=actual_temp
                ))
            except Exception as e:
                yield f"data: {json.dumps({'word': f'\n[Generation Initialization Error: {str(e)}]', 'expert_id': 0, 'expert_name': 'System'})}\n\n"
                return

            final_text = ""
            while True:
                try:
                    token = await asyncio.to_thread(next, iterator)
                    if token is None: break
                    final_text += token
                    
                    # Stop if model generates the EOS token or a stop sequence
                    if token in ["<|endoftext|>", "<|im_end|>", "</s>", "unknown"]:
                        break
                        
                    payload = json.dumps({
                        'word': token,
                        'expert_id': 0,
                        'expert_name': display_name,
                        'confidence': 0.0
                    })
                    yield f"data: {payload}\n\n"
                except StopIteration:
                    break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    error_msg = json.dumps({
                        'word': f"\n[Backend Iterator Error: {str(e)}]",
                        'expert_id': 0,
                        'expert_name': 'System',
                        'confidence': 0.0
                    })
                    yield f"data: {error_msg}\n\n"
                    break
            
            if not final_text.strip():
                # If we got here with nothing, maybe the model is still loading weights internally or OOM
                error_msg = json.dumps({
                    'word': "\n[Backend yielded an empty response. This often happens if the model is still downloading or if GPU memory is full.]",
                    'expert_id': 0,
                    'expert_name': 'System',
                    'confidence': 0.0
                })
                yield f"data: {error_msg}\n\n"
            
            orch = get_orchestrator()
            if orch:
                orch.record_chat_interaction(req.message, final_text, 0, 1.0)
            return

        # Original Custom MoE Logic
        combined_input = f"{full_persona}\n\nUser: {user_query}\nAnswer:"
        tokenized_input = tokenize(combined_input)
        ids = torch.tensor([[get_word_id(w) for w in tokenized_input[-CONTEXT_LEN:]]]).long().to(DEVICE)
        model.eval()
        
        # Initialization packet
        yield f"data: {json.dumps({'word': '', 'expert_id': 0, 'expert_name': 'MoE Core', 'confidence': 0.0})}\n\n"
        
        final_text = ""
        avg_confidence = 0.0
        last_expert_id = 0
        with torch.no_grad():
            for i in range(100): # Max stream length
                outputs, expert_id_tensor = model(ids)
                expert_id = expert_id_tensor.item()
                last_expert_id = expert_id
                
                logits = outputs[0, -1, :] if outputs.dim() == 3 else outputs[0]
                actual_temp = max(req.temperature, 0.01)
                probs = torch.softmax(logits / actual_temp, dim=-1)
                
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
                
                payload = json.dumps({
                    'word': word + " ", # Add space for word-based tokenizer
                    'expert_id': expert_id,
                    'expert_name': DOMAINS[expert_id] if expert_id < len(DOMAINS) else "general",
                    'confidence': conf
                })
                yield f"data: {payload}\n\n"
                
                if word in [".", "!", "?", "<eos>"] and i > 5:
                    break
                    
                new_id = torch.tensor([[predicted_id]]).long().to(DEVICE)
                ids = torch.cat([ids, new_id], dim=1)[:, -CONTEXT_LEN:]
                await asyncio.sleep(0.01)
                
        orch = get_orchestrator()
        if orch:
            orch.record_chat_interaction(req.message, final_text.strip(), last_expert_id, avg_confidence)

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    get_persistence_cached().increment_stat("total_interactions")
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
    if isinstance(req.message, str):
        req.message = tokenize(req.message)
    elif isinstance(req.message, list):
        if len(req.message) == 1 and " " in req.message[0]:
             req.message = tokenize(req.message[0])
    
    context_str = ""
    retrieved_chunks = []
    if req.use_rag:
        try:
            retrieved_chunks = await get_rag().retrieve(req.message, k=5, use_web=req.use_web)
            if retrieved_chunks:
                context_str = "\n\nContext information:\n" + "\n".join([" ".join(c) for c in retrieved_chunks])
        except Exception as e:
            print(f"RAG Retrieval Error: {e}")
    
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
        # Prepend persona and context for custom MoE
        combined_prompt = f"{full_persona}\n\nUser: {user_query}\nAnswer:"
        gen_result = await asyncio.to_thread(
            generate_text,
            model, vocab, combined_prompt,
            max_new_words=req.max_tokens, temperature=req.temperature, top_k=req.top_k, context_len=CONTEXT_LEN
        )
        final_response = gen_result["text"]
        avg_confidence = gen_result["avg_confidence"]
        main_expert_id = gen_result["main_expert_id"]
    
    # Record for learning (using the full generated text as feedback for next time)
    orch = get_orchestrator()
    if orch:
        orch.record_chat_interaction(req.message, final_response, main_expert_id, avg_confidence)
    
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
        get_rag().save()
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
    
    # Calculate performance metrics
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    
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

    # Performance Diagnosis
    perf_status = "Excellent"
    if DEVICE.type == 'cpu' and USE_EXTERNAL_MODEL:
        perf_status = "Critical (7B on CPU is unsupported for real-time)"
    elif DEVICE.type == 'cpu':
        perf_status = "Reduced (Custom MoE on CPU)"
        
    return {
        "expert_utilization": processed_util,
        "expert_names": expert_names,
        "vocab_size": v_size,
        "num_experts": n_experts,
        "device": str(DEVICE),
        "is_external": is_ext,
        "model_name": model_name,
        "status": "active",
        "perf_status": perf_status,
        "cpu_load": cpu_usage,
        "ram_load": ram_usage,
        "restore_info": restore_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
