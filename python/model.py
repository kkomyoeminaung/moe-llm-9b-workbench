import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, emb_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
    def forward(self, x):
        return self.net(x)

class Expert(nn.Module):
    def __init__(self, emb_dim, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )
    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, emb_dim=768, num_experts=10):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.router = Router(emb_dim, num_experts)
        self.experts = nn.ModuleList([Expert(emb_dim) for _ in range(num_experts)])

    def forward(self, x):
        batch_size = x.size(0)
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        # Top-1 gating
        routing_probs, selected_experts = torch.max(probs, dim=-1)
        
        # Improved Load balancing loss (KL divergence)
        # Target: uniform distribution (1/num_experts for each)
        target_prob = torch.ones(self.num_experts, device=x.device) / self.num_experts
        actual_prob = torch.mean(probs, dim=0) # [num_experts]
        
        # KL divergence loss KL(P || Q) = kl_div(log(Q), P)
        load_balancing_loss = F.kl_div(
            target_prob.log(),
            actual_prob.clamp(min=1e-8),
            reduction='batchmean'
        ) * self.num_experts # Scale to meaningful range
        
        # Expert forward
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (selected_experts == i)
            if mask.any():
                output[mask] = self.experts[i](x[mask])
                
        return output, load_balancing_loss
