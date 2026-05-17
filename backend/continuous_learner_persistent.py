# backend/continuous_learner_persistent.py
"""Continuous learner with auto-persistence"""

from backend.persistence_auto import get_persistence
from collections import deque
from typing import List, Dict

class PersistentContinuousLearner:
    """Continuous learner that persists to storage"""
    
    def __init__(self, capacity: int = 10000):
        self.persistence = get_persistence()
        self.capacity = capacity
        
        self.memory = deque(maxlen=capacity)
        # Load from persistence
        saved_memory = self.persistence.load_metadata("learner_memory", [])
        for item in saved_memory:
            self.memory.append(item)
            
        self.step = self.persistence.load_setting("learner_step", 0)
        
        print(f"🧠 Continuous learner initialized: {len(self.memory)} memories loaded.")
    
    def store_episode(self, input_words: List[str], output_word: str, 
                         expert_id: int, confidence: float):
        """Store experience with auto-persistence"""
        experience = {
            "input": input_words,
            "output": output_word,
            "expert": expert_id,
            "step": self.step,
            "confidence": confidence
        }
        
        self.memory.append(experience)
        self.step += 1
        
        # Auto-save periodically
        if len(self.memory) % 50 == 0:
            self.persistence.save_metadata("learner_memory", list(self.memory))
            self.persistence.save_setting("learner_step", self.step)
            print(f"💾 Persisted {len(self.memory)} memories.")
    
    def get_stats(self) -> dict:
        return {
            "total_memories": len(self.memory),
            "step": self.step,
            "persisted": True
        }
