# backend/persistence_auto.py
import os
import json
import pickle
import sqlite3
try:
    import faiss
except ImportError:
    print("⚠️ FAISS not installed. RAG will be disabled.")
    faiss = None
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import atexit

class AutoPersistence:
    def __init__(self, project_name: str = "moe-llm"):
        self.project_name = project_name
        
        # Detect environment
        self.is_lightning = bool(os.environ.get('LIGHTNING_CLOUD', False))
        self.is_kaggle = os.path.exists('/kaggle/working')
        self.is_colab = os.path.exists('/content')
        
        # Setup paths
        if self.is_lightning:
            self.data_dir = Path("/teamspace/studios/this_studio/data")
        elif self.is_kaggle:
            self.data_dir = Path("/kaggle/working/data")
        elif self.is_colab:
            if os.path.exists('/content/drive'):
                self.data_dir = Path(f"/content/drive/MyDrive/{project_name}")
            else:
                self.data_dir = Path("/content/data")
        else:
            self.data_dir = Path("data")
            
        self.rag_dir = self.data_dir / "rag"
        self.memory_dir = self.data_dir / "memory"
        self.checkpoint_dir = self.data_dir / "checkpoints"
        
        # Setup directories (Auto-create)
        for d in [self.data_dir, self.rag_dir, self.memory_dir, self.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Initialize DB
        self.db_path = self.data_dir / "moe.db"
        self._init_database()
        
        atexit.register(self.save_all)
        print("✅ AutoPersistence initialized and ready.")

    @contextmanager
    def transaction(self):
        """Context manager for database operations"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            yield conn.cursor()
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self):
        with self.transaction() as cursor:
            cursor.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
            cursor.execute('CREATE TABLE IF NOT EXISTS checkpoints (id INTEGER PRIMARY KEY, step INTEGER, path TEXT, loss REAL)')
            cursor.execute('CREATE TABLE IF NOT EXISTS stats (id INTEGER PRIMARY KEY, total_interactions INTEGER DEFAULT 0, rag_chunks INTEGER DEFAULT 0, memory_size INTEGER DEFAULT 0)')
            cursor.execute('INSERT OR IGNORE INTO stats (id, total_interactions) VALUES (1, 0)')

    def save_rag(self, index, chunks):
        try:
            faiss.write_index(index, str(self.rag_dir / "faiss.index"))
            with open(self.rag_dir / "chunks.pkl", 'wb') as f:
                pickle.dump(chunks, f)
            print("💾 RAG index saved.")
        except Exception as e:
            print(f"⚠️ RAG save failed: {e}")

    def load_rag(self):
        index_path = self.rag_dir / "faiss.index"
        chunks_path = self.rag_dir / "chunks.pkl"
        if index_path.exists() and chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            return faiss.read_index(str(index_path)), chunks
        return None, []

    def save_checkpoint(self, model_state, step, loss):
        path = self.checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save({'step': step, 'model_state': model_state, 'loss': loss}, path)
        with self.transaction() as cursor:
            cursor.execute("INSERT INTO checkpoints (step, path, loss) VALUES (?, ?, ?)", (step, str(path), loss))
    
    def save_metadata(self, key: str, value: Any):
        with self.transaction() as cursor:
            cursor.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', (key, json.dumps(value)))

    def load_metadata(self, key: str, default: Any = None) -> Any:
        try:
            with self.transaction() as cursor:
                cursor.execute('SELECT value FROM metadata WHERE key = ?', (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return default

    def save_setting(self, key: str, value: Any):
        self.save_metadata(f"setting_{key}", value)

    def load_setting(self, key: str, default: Any = None) -> Any:
        return self.load_metadata(f"setting_{key}", default)

    def increment_stat(self, stat_name: str, amount: int = 1):
        try:
            with self.transaction() as cursor:
                cursor.execute(f"UPDATE stats SET {stat_name} = {stat_name} + ?", (amount,))
        except Exception as e:
            # Fallback if column doesn't exist
            print(f"⚠️ Failed to increment stat {stat_name}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Fetch current stats from database"""
        try:
            with self.transaction() as cursor:
                cursor.execute("SELECT total_interactions, rag_chunks, memory_size FROM stats WHERE id=1")
                row = cursor.fetchone()
                if row:
                    return {
                        "total_interactions": row[0],
                        "rag_chunks": row[1],
                        "memory_size": row[2]
                    }
        except Exception:
            pass
        return {"total_interactions": 0, "rag_chunks": 0, "memory_size": 0}

    def save_all(self):
        # Already committed via transactions
        print("✅ Data persistence ensured.")

_persistence_instance = None
def get_persistence():
    global _persistence_instance
    if _persistence_instance is None: 
        _persistence_instance = AutoPersistence()
    return _persistence_instance
