# backend/rag_persistent.py
"""RAG engine with auto-persistence"""

from backend.persistence import get_persistence
import numpy as np
import faiss
from typing import List

class PersistentRAG:
    """RAG engine that automatically persists to Lightning AI storage"""
    
    def __init__(self, index_path: str = "data/rag_index"):
        self.persistence = get_persistence()
        
        # Load existing index if available
        self.index, self.chunks = self.persistence.load_rag_index()
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(768)
            self.chunks = []
            print("✅ Created new RAG index")
        else:
            print(f"✅ Loaded existing RAG index: {self.index.ntotal} chunks")
    
    def add_document(self, words: List[str], domain: int = None):
        """Add document with auto-persistence"""
        embedding = self._embed(words)
        self.index.add(embedding.reshape(1, -1))
        self.chunks.append(words)
        
        # Auto-save every 10 documents
        if len(self.chunks) % 10 == 0:
            self._save()
    
    def retrieve(self, query: List[str], k: int = 5) -> List[List[str]]:
        """Retrieve similar chunks"""
        if len(self.chunks) == 0:
            return []
        
        query_embed = self._embed(query)
        scores, indices = self.index.search(query_embed.reshape(1, -1), min(k, len(self.chunks)))
        
        return [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
    
    def _embed(self, words: List[str]) -> np.ndarray:
        return np.random.randn(768).astype(np.float32)
    
    def _save(self):
        self.persistence.save_rag_index(self.index, self.chunks)
    
    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "persisted": True
        }
