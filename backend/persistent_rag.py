# backend/persistent_rag.py
"""FAISS with disk persistence"""

import faiss
import pickle
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict

class PersistentRAG:
    """RAG with disk persistence using FAISS + SQLite"""
    
    def __init__(self, index_path: str = "data/rag_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Load embedding model
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embed_dim = self.encoder.get_sentence_embedding_dimension()
        
        # FAISS index file
        self.index_file = self.index_path / "faiss.index"
        self.chunks_file = self.index_path / "chunks.pkl"
        self.metadata_db = self.index_path / "metadata.db"
        
        # Load or create FAISS index
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatIP(self.embed_dim)
            self.chunks = []
            print("✅ Created new FAISS index")
        
        # SQLite for metadata
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite metadata database"""
        self.conn = sqlite3.connect(str(self.metadata_db), check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                domain INTEGER,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()
    
    def add_document(self, words: List[str], domain: int = None, source: str = "user"):
        """Add document to RAG with persistence"""
        # Create embedding
        text = ' '.join(words) if isinstance(words, list) else words
        embedding = self.encoder.encode([text]).astype('float32')
        faiss.normalize_L2(embedding)
        
        # Add to FAISS
        self.index.add(embedding)
        self.chunks.append(words)
        
        # Add to SQLite metadata
        self.cursor.execute(
            "INSERT INTO chunks (content, domain, source) VALUES (?, ?, ?)",
            (text, domain, source)
        )
        self.conn.commit()
        
        # Save to disk
        self._save()
    
    def retrieve(self, query: List[str], k: int = 5) -> List[List[str]]:
        """Retrieve similar chunks"""
        if len(self.chunks) == 0:
            return []
        
        query_text = ' '.join(query) if isinstance(query, list) else query
        query_emb = self.encoder.encode([query_text]).astype('float32')
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, min(k, len(self.chunks)))
        
        # Update access count
        for idx in indices[0]:
            if idx < len(self.chunks) and idx >= 0:
                self.cursor.execute(
                    "UPDATE chunks SET access_count = access_count + 1 WHERE id = ?",
                    (int(idx) + 1,)
                )
        self.conn.commit()
        
        return [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks) and idx >= 0]
    
    def _save(self):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, str(self.index_file))
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"💾 Saved FAISS index: {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        self.cursor.execute("SELECT COUNT(*), COUNT(DISTINCT domain), SUM(access_count) FROM chunks")
        result = self.cursor.fetchone()
        count, domains, total_access = result if result else (0, 0, 0)
        
        return {
            "total_chunks": count,
            "unique_domains": domains,
            "total_accesses": total_access or 0,
            "faiss_vectors": self.index.ntotal
        }
