# backend/database.py
"""Unified database layer for Level 5"""

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.persistent_rag import PersistentRAG

class MoEDatabase:
    """Unified database layer combining all storage needs"""
    
    def __init__(self, base_path: str = "data"):
        from backend.persistent_rag import PersistentRAG
        from backend.persistent_memory import PersistentEpisodicMemory
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rag = PersistentRAG(str(self.base_path / "rag"))
        self.memory = PersistentEpisodicMemory(str(self.base_path / "memory.db"))
        
        # Main metadata database
        self.metadata_db = sqlite3.connect(str(self.base_path / "metadata.db"), check_same_thread=False)
        self._init_metadata()
        print("✅ Database layer initialized")
    
    def _init_metadata(self):
        """Initialize main metadata tables"""
        cursor = self.metadata_db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.metadata_db.commit()
    
    def get_full_stats(self) -> dict:
        """Get complete database statistics"""
        return {
            "rag": self.rag.get_stats(),
            "memory": self.memory.get_stats()
        }

    def close(self):
        if hasattr(self, 'metadata_db'):
            self.metadata_db.close()
        if hasattr(self, 'memory'):
            self.memory.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass
