# backend/rag_engine.py
import torch
import faiss
import numpy as np
from typing import List, Dict

class RAGEngine:
    def __init__(self, embed_dim=384):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embed_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embed_dim)  # Inner product
        self.docs = []
        
    def add_document(self, chunk: List[str], domain: int = 0, source: str = "file"):
        text = ' '.join(chunk) if isinstance(chunk, list) else chunk
        embedding = self.encoder.encode([text]).astype('float32')
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.docs.append({"chunk": chunk, "domain": domain, "source": source})
        
    def retrieve(self, query_words: List[str], k=3) -> List[List[str]]:
        if self.index.ntotal == 0:
            return []
        query = ' '.join(query_words)
        query_emb = self.encoder.encode([query]).astype('float32')
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.docs):
                results.append(self.docs[idx]["chunk"])
        return results
