import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Expert(nn.Module):
    """0.5B parameter expert with gradient checkpointing"""
    def __init__(self, expert_id: int, embed_dim: int = 768, ff_dim: int = 3072, num_layers: int = 6, vocab_size: int = 50000):
        super().__init__()
        self.expert_id = expert_id
        self.use_checkpointing = True
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=ff_dim,
                activation='gelu',
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, vocab_size)  # configurable vocab size
    
    def _forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, is_causal=True)
        return self.output_proj(x)
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class SparseMoE(nn.Module):
    """Switch Transformer with Top-1 routing, load balancing, and positional encoding"""
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 768, 
                 num_experts: int = 10, max_len: int = 2048):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        # Embedding layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        
        # Router (Switch Transformer)
        self.router = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts)
        )
        
        # 10 experts × 0.5B = 5B total
        self.experts = nn.ModuleList([
            Expert(i, embed_dim) for i in range(num_experts)
        ])
        
        # Load balancing tracking
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('global_step', torch.tensor(0))
    
    def forward(self, x, return_router_probs=False):
        """
        x: [batch, seq_len] word indices
        returns: logits [batch, vocab_size], expert_ids [batch], router_probs [batch, num_experts]
        """
        # Embed + positional encoding
        x = self.embed(x)  # [batch, seq_len, embed_dim]
        x = self.pos_encoder(x)
        
        # Router: average over sequence then route
        seq_avg = x.mean(dim=1)  # [batch, embed_dim]
        router_logits = self.router(seq_avg)  # [batch, num_experts]
        
        # Add noise during training to prevent collapse
        if self.training:
            noise = torch.randn_like(router_logits) * 0.01
            router_logits = router_logits + noise
        
        router_probs = F.softmax(router_logits, dim=-1)
        top1_probs, top1_indices = router_probs.max(dim=-1)  # [batch]
        
        # Update usage statistics
        for i in range(self.num_experts):
            self.expert_usage[i] = (top1_indices == i).sum().item()
        
        # Sparse forward: only active expert processes each sample
        outputs = torch.zeros(x.size(0), self.vocab_size, device=x.device)
        
        for expert_id in range(self.num_experts):
            mask = (top1_indices == expert_id)
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[expert_id](expert_input)
                # Take last token's output for next word prediction
                outputs[mask] = expert_output[:, -1, :]
        
        if return_router_probs:
            return outputs, top1_indices, router_probs
        return outputs, top1_indices
    
    def get_load_balancing_loss(self):
        """Auxiliary loss to ensure experts are used equally"""
        # Target: uniform distribution
        num_tokens = self.expert_usage.sum()
        if num_tokens == 0:
            return torch.tensor(0.0, device=self.expert_usage.device)
        
        target_prob = torch.ones(self.num_experts, device=self.expert_usage.device) / self.num_experts
        actual_prob = self.expert_usage / num_tokens
        
        # KL divergence loss KL(P || Q) = kl_div(log(Q), P)
        kl_loss = F.kl_div(
            target_prob.log(),
            actual_prob.clamp(min=1e-8),
            reduction='batchmean'
        )
        
        # Reset for next batch
        self.expert_usage.zero_()
        self.global_step += 1
        
        return kl_loss
    
    def get_expert_utilization(self):
        """Return utilization percentage for each expert"""
        total = self.expert_usage.sum()
        if total == 0:
            return torch.zeros(self.num_experts)
        return self.expert_usage / total
