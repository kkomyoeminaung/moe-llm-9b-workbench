# backend/persistence.py
"""Lightning AI auto-persistence handler"""

import os
import json
import pickle
import shutil
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import sqlite3

class LightningPersistence:
    """
    Automatic data persistence for Lightning AI
    Uses lit:// drive for cloud storage, falls back to local
    """
    
    def __init__(self, project_name: str = "moe-llm"):
        self.project_name = project_name
        
        # Detect environment
        self.is_lightning = os.environ.get('LIGHTNING_CLOUD', False)
        self.is_kaggle = os.path.exists('/kaggle/working')
        self.is_colab = os.path.exists('/content')
        
        # Setup paths
        if self.is_lightning:
            # Lightning AI cloud storage
            self.base_path = Path(f"lit://{project_name}")
            self.local_cache = Path("/teamspace/studios/this_studio/cache")
        elif self.is_kaggle:
            # Kaggle working directory is persistent if saved
            self.base_path = Path("/kaggle/working/data")
            self.local_cache = Path("/kaggle/working/data/cache")
        elif self.is_colab:
            # If Google Drive is mounted, we prefer it
            if os.path.exists('/content/drive'):
                self.base_path = Path(f"/content/drive/MyDrive/{project_name}")
            else:
                self.base_path = Path("/content/data")
            self.local_cache = Path("/content/data/cache")
        else:
            # Local development
            self.base_path = Path("data")
            self.local_cache = Path("data/cache")
        
        # Create directories
        self.local_cache.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.checkpoints_dir = self.base_path / "checkpoints"
        self.models_dir = self.base_path / "models"
        self.database_dir = self.base_path / "database"
        self.datasets_dir = self.base_path / "datasets"
        
        # Initialize
        self._init_directories()
        self._init_metadata()
        
        print(f"💾 LightningPersistence initialized. Base path: {self.base_path}")
    
    def _init_directories(self):
        """Create directories if they don't exist"""
        for dir_path in [self.checkpoints_dir, self.models_dir, 
                         self.database_dir, self.datasets_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except:
                pass
    
    def _init_metadata(self):
        """Initialize SQLite metadata database"""
        self.metadata_path = self.database_dir / "metadata.db"
        
        # Use local cache for SQLite
        if self.is_lightning:
            local_metadata = self.local_cache / "metadata.db"
            if local_metadata.exists() and not self.metadata_path.exists():
                shutil.copy(local_metadata, self.metadata_path)
        
        self.conn = sqlite3.connect(str(self.metadata_path))
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                path TEXT,
                loss REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def save_checkpoint(self, model_state: dict, step: int, loss: float):
        """Save model checkpoint automatically"""
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{step}.pt"
        
        checkpoint = {
            'step': step,
            'model_state': model_state,
            'loss': loss,
        }
        
        torch.save(checkpoint, str(checkpoint_path))
        # Record in database
        self.cursor.execute(
            "INSERT INTO checkpoints (step, path, loss) VALUES (?, ?, ?)",
            (step, str(checkpoint_path), loss)
        )
        self.conn.commit()
        return str(checkpoint_path)
    
    def save_rag_index(self, index, chunks):
        """Save FAISS RAG index"""
        import faiss
        index_path = self.database_dir / "faiss.index"
        chunks_path = self.database_dir / "chunks.pkl"
        
        faiss.write_index(index, str(index_path))
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
    
    def load_rag_index(self):
        """Load FAISS RAG index"""
        import faiss
        index_path = self.database_dir / "faiss.index"
        chunks_path = self.database_dir / "chunks.pkl"
        
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            return index, chunks
        return None, None
    
    def save_metadata(self, key: str, value: Any):
        """Save metadata"""
        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, json.dumps(value))
        )
        self.conn.commit()
    
    def load_metadata(self, key: str, default=None):
        """Load metadata"""
        self.cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = self.cursor.fetchone()
        if row:
            return json.loads(row[0])
        return default
    
    def sync_to_cloud(self):
        """Sync local cache to cloud if needed"""
        pass

_persistence = None
def get_persistence() -> LightningPersistence:
    """Get global persistence instance"""
    global _persistence
    if _persistence is None:
        _persistence = LightningPersistence()
    return _persistence
