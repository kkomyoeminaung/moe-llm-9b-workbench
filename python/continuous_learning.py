import random
import torch
import torch.nn.functional as F

class EpisodicMemory:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, sample):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, n=100):
        if not self.buffer: return []
        return random.sample(self.buffer, min(n, len(self.buffer)))

def sleep_consolidation(model, memory_buffer, optimizer):
    """Replay process to prevent catastrophic forgetting"""
    if not memory_buffer.buffer:
        return
        
    samples = memory_buffer.sample(100)
    model.train()
    
    total_loss = 0
    for sample in samples:
        input_ids = torch.tensor([sample['input']]).long()
        targets = torch.tensor([sample['target']]).long()
        
        outputs, lb_loss = model(input_ids)
        loss = F.cross_entropy(outputs, targets) + 0.01 * lb_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        
    print(f"😴 Consolidating memory: Avg Loss = {total_loss/len(samples):.4f}")
    model.eval()
