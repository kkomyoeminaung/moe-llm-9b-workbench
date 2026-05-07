# training/train_with_persistence.py
"""Training with automatic checkpointing and persistence"""

import torch
import torch.nn as nn
from backend.persistence import get_persistence
from backend.rag_persistent import PersistentRAG
from backend.continuous_learner_persistent import PersistentContinuousLearner

def train_with_persistence():
    """Training function with auto-persistence"""
    
    # Initialize persistence
    persistence = get_persistence()
    
    print("🚀 Training with auto-persistence initiated")
    
    # Initialize components
    rag = PersistentRAG()
    learner = PersistentContinuousLearner()
    
    # Training simulation loop
    for step in range(1001):
        # Save checkpoint every 500 steps
        if step % 500 == 0:
            persistence.save_checkpoint(
                {"dummy": "state"},
                step,
                0.1
            )
            print(f"💾 Step {step}: Checkpoint saved")
            
    print("✅ Training complete with auto-persistence!")

if __name__ == "__main__":
    train_with_persistence()
