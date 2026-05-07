# training/model_unified.py - Works on CPU and GPU
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from config import DEVICE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=12, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Expert(nn.Module):
    def __init__(self, expert_id, embed_dim=768, num_layers=6, ff_dim=3072, vocab_size=50000):
        super().__init__()
        self.expert_id = expert_id
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ff_dim=ff_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.use_checkpointing = False  # Set True for GPU training
    
    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                # Use reentrant=False is safer with newer PyTorch
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.output_proj(x[:, -1, :])


class SparseMoE_Unified(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=768, num_experts=10, 
                 max_len=2048, expert_layers=6, ff_dim=3072):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.max_len = max_len
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        
        # Router (Switch Transformer)
        self.router = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts)
        )
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(i, embed_dim, num_layers=expert_layers, ff_dim=ff_dim) 
            for i in range(num_experts)
        ])
        
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.to(DEVICE)
    
    def forward(self, x, return_details=False):
        batch_size = x.size(0)
        actual_len = min(x.size(1), self.max_len)
        x = x[:, :actual_len]
        
        # Embed + position
        x = self.embed(x)
        x = self.pos_encoder(x)
        
        # Router implementation - Using last token for sequence awareness
        x_router = x[:, -1, :]
        router_logits = self.router(x_router)
        
        # Add noise during training
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.01
        
        router_probs = F.softmax(router_logits, dim=-1)
        top1_probs, top1_indices = router_probs.max(dim=-1)
        
        # Update usage (use device safe operations)
        with torch.no_grad():
            for i in range(self.num_experts):
                self.expert_usage[i] += (top1_indices == i).sum().item()
        
        # Sparse forward
        outputs = torch.zeros(batch_size, self.vocab_size, device=x.device)
        
        for expert_id in range(self.num_experts):
            mask = (top1_indices == expert_id)
            if mask.any():
                expert_input = x[mask]
                # Scale expert output by router probability for proper gradients
                expert_output = self.experts[expert_id](expert_input) * top1_probs[mask].unsqueeze(-1)
                outputs[mask] = expert_output
        
        if return_details:
            return outputs, top1_indices, router_probs
        return outputs, top1_indices
    
    def get_load_balancing_loss(self):
        total = self.expert_usage.sum()
        if total == 0:
            return torch.tensor(0.0, device=self.expert_usage.device)
        
        target = torch.ones(self.num_experts, device=self.expert_usage.device) / self.num_experts
        actual = (self.expert_usage / total).clamp(min=1e-8)
        # Using actual.log() for KL(actual || target) or F.kl_div correctly
        kl_loss = F.kl_div(actual.log(), target, reduction='batchmean')
        self.expert_usage.zero_()
        return kl_loss
    
    def get_expert_utilization(self):
        total = self.expert_usage.sum()
        if total == 0:
            return [0] * self.num_experts
        return (self.expert_usage / total).tolist()
    
    def to_device_optimized(self):
        """Auto-optimize for current device"""
        if DEVICE.type == 'cuda':
            # GPU optimizations
            for expert in self.experts:
                expert.use_checkpointing = True
            print("🚀 GPU optimizations enabled (checkpointing)")
        else:
            # CPU optimizations
            for expert in self.experts:
                expert.use_checkpointing = False
            # Reduce threads for better CPU performance
            torch.set_num_threads(min(torch.get_num_threads(), 8))
            print(f"🖥️ CPU optimizations enabled ({torch.get_num_threads()} threads)")
        
        return self
