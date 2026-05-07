# backend/model_loader.py
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add training folder to path to import model and config
sys.path.append(str(Path(__file__).parent.parent / "training"))
from config import DEVICE, VOCAB_SIZE, EMBED_DIM, NUM_EXPERTS, EXPERT_LAYERS, HIDDEN_DIM, CONTEXT_LEN
from model_unified import SparseMoE_Unified

class MoELoader(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        # 1. Initialize Real Architecture (same as training)
        self.model = SparseMoE_Unified(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            num_experts=NUM_EXPERTS,
            max_len=CONTEXT_LEN,
            expert_layers=EXPERT_LAYERS,
            ff_dim=HIDDEN_DIM
        ).to_device_optimized()
        self.model.eval()
        
        # 2. Strategy to find the best model locally
        if model_path is None:
            options = [
                "checkpoints/best.pt",
                "checkpoints/moe_final.pt",
                "checkpoints/moe_model_complete.pt",
                "models/moe_5b_final_cpu.pt"
            ]
            for opt in options:
                if Path(opt).exists():
                    model_path = opt
                    break
        
        if model_path:
            self.load_weights(model_path)

    def load_weights(self, path):
        print(f"📥 Loading backend weights from {path}...")
        try:
            state_dict = torch.load(path, map_location=DEVICE)
            # Remove prefix if present
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ Backend Model Ready.")
        except Exception as e:
            print(f"⚠️ Load failed: {e}")

    def forward(self, word_ids):
        # SparseMoE_Unified forward returns (outputs, expert_indices)
        return self.model(word_ids)

    def get_expert_utilization(self):
        return self.model.get_expert_utilization()

# Global singleton
_model = None

def get_model():
    global _model
    if _model is None:
        _model = MoELoader()
    return _model
