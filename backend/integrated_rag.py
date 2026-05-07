# backend/integrated_rag.py
"""Complete RAG with local FAISS + Web Search + Auto-Persistence"""

import asyncio
from typing import List, Optional
import numpy as np
from backend.persistence_auto import get_persistence
from backend.web_search_rag import WebSearchRAG

class IntegratedRAG:
    """Integrated RAG system"""
    
    def __init__(self, embed_dim: int = 384): # all-MiniLM-L6-v2 is 384
        self.persistence = get_persistence()
        self.embed_dim = embed_dim
        
        # Try to load semantic encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("🧠 Semantic Encoder Loaded (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"⚠️ Semantic Encoder failed to load: {e}. Falling back to hashing.")
            self.encoder = None

        # Load or create FAISS index
        self.index, self.chunks = self.persistence.load_rag()
        if self.index is None:
            import faiss
            self.index = faiss.IndexFlatIP(embed_dim)
            self.chunks = []
            print(f"✅ Created new FAISS index (IP, dim={embed_dim})")
        
        # Initialize web search
        self.web_search = WebSearchRAG(self)
        
        print(f"✅ IntegratedRAG initialized: {len(self.chunks)} local chunks")
    
    def _embed(self, words: List[str]) -> np.ndarray:
        """Create embedding (Semantic if available, else deterministic)"""
        text = ' '.join(words)
        
        if self.encoder:
            try:
                embedding = self.encoder.encode([text])[0]
                return embedding.astype(np.float32)
            except:
                pass

        # Fallback Hashing
        import hashlib
        embedding = np.zeros(self.embed_dim)
        for i, word in enumerate(words[:self.embed_dim]):
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            embedding[i % self.embed_dim] += (h % 100) / 100.0
        return embedding.astype(np.float32)
    
    def add_document(self, words: List[str], domain: int = None, source: str = "user"):
        """Add document to local RAG"""
        if not words:
            return
        
        embedding = self._embed(words)
        self.index.add(embedding.reshape(1, -1))
        self.chunks.append({
            "words": words,
            "domain": domain,
            "source": source
        })
        
        # Auto-save every 10 documents
        if len(self.chunks) % 10 == 0:
            self.persistence.save_rag(self.index, self.chunks)
            self.persistence.increment_stat("rag_chunks", 10)
    
    def retrieve_local(self, query: List[str], k: int = 5) -> List[List[str]]:
        """Retrieve from local FAISS index only"""
        if len(self.chunks) == 0:
            return []
        
        query_embed = self._embed(query)
        scores, indices = self.index.search(query_embed.reshape(1, -1), min(k, len(self.chunks)))
        
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append(chunk["words"])
        
        return results
    
    async def retrieve(self, query: List[str], k: int = 5, use_web: bool = True) -> List[List[str]]:
        """Retrieve from both local and web"""
        results = []
        
        # 1. Local retrieval
        local_results = self.retrieve_local(query, k=min(k, 3))
        results.extend(local_results)
        
        # 2. Web search if needed
        if use_web and len(results) < k:
            web_results = await self.web_search.retrieve_with_web(
                query, use_web=True, k=k - len(results)
            )
            results.extend(web_results)
        
        return results[:k]
    
    def save(self):
        """Manually save RAG"""
        self.persistence.save_rag(self.index, self.chunks)
    
    async def close(self):
        """Close web search session"""
        await self.web_search.close()
        self.save()
