# backend/persistent_memory.py
"""Disk-persistent episodic memory for continuous learning"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict

class PersistentEpisodicMemory:
    """SQLite-based persistent memory for continuous learning"""
    
    def __init__(self, db_path: str = "data/memory.db"):
        import threading
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set check_same_thread=False for async frameworks like FastAPI
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._lock = threading.Lock()
        
        self._init_tables()
    
    def _init_tables(self):
        """Initialize all tables"""
        # Episodes table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                output_text TEXT,
                expert_id INTEGER,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Corrections table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                incorrect_output TEXT,
                correct_output TEXT,
                expert_id INTEGER,
                applied BOOLEAN DEFAULT 0
            )
        ''')
        self.conn.commit()
    
    def store_episode(self, input_words: List[str], output: str, 
                      expert_id: int, confidence: float):
        """Store interaction episode"""
        with self._lock:
            self.cursor.execute('''
                INSERT INTO episodes (input_text, output_text, expert_id, confidence)
                VALUES (?, ?, ?, ?)
            ''', (' '.join(input_words), output, expert_id, confidence))
            self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        with self._lock:
            self.cursor.execute("SELECT COUNT(*) FROM episodes")
            total = self.cursor.fetchone()[0]
        return {"total_episodes": total}

    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass
