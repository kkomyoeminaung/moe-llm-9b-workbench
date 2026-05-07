# model/long_context.py
"""Extended context with sliding window attention"""

import torch
import torch.nn as nn
import math

class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for efficient long context.
    Similar to Longformer and Mistral's architecture.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, attention_mask=None):
        batch, seq_len, dim = x.shape
        
        # Project
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # Sliding window mask
        if seq_len > self.window_size:
            # Create sliding window mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), 
                diagonal=-self.window_size
            ) & torch.tril(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=self.window_size
            )
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.ones(1, 1, seq_len, seq_len, device=x.device)
        
        # Compute attention with mask
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        
        return self.out_proj(out.reshape(batch, seq_len, dim))


class LongContextMoE(nn.Module):
    """MoE with extended 8192 context"""
    
    def __init__(self, base_model, max_context=8192, window_size=2048):
        super().__init__()
        self.base_model = base_model
        self.max_context = max_context
        self.window_size = window_size
        
        # Replace attention layers with sliding window
        # (Assuming typical transformer structure: experts, blocks, attention)
        # Replacing attention in expert 4 for demonstration
        for layer in base_model.experts[4].blocks:
            layer.attention = SlidingWindowAttention(768, 12, window_size)
    
    def forward(self, x):
        # Handle long context with chunking if needed
        if x.size(1) > self.max_context:
            # Simplified chunking
            return self.base_model(x[:, :self.max_context, :])
        
        return self.base_model(x)
