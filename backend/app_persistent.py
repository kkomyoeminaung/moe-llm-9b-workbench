# backend/app_persistent.py
"""FastAPI app with auto-persistence"""

from fastapi import FastAPI
from backend.persistence import get_persistence
from backend.rag_persistent import PersistentRAG
from backend.continuous_learner_persistent import PersistentContinuousLearner

app = FastAPI()

# Initialize with persistence
persistence = get_persistence()
rag = PersistentRAG()
learner = PersistentContinuousLearner()

@app.on_event("startup")
async def startup():
    """Load persisted data on startup"""
    print("🔄 Loading persisted data...")
    print("✅ Persistence layers ready.")

@app.get("/stats/persistence")
async def persistence_stats():
    """Get persistence statistics"""
    return persistence.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
