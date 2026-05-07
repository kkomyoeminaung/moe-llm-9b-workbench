# training/train_lightning.py
import os
import sys
import json
import torch
import torch.nn as nn
import time
import random
from pathlib import Path
from backend.persistence_auto import get_persistence

# Setup
persistence = get_persistence()
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("data/checkpoints")

# DataLoader & Model
dataset_path = DATA_DIR / "train.jsonl"
vocab_path = DATA_DIR / "vocab.json"

# Auto-generate if missing
if not dataset_path.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = [{"sentence": [f"w{i}"], "next_word": "w1"} for i in range(1000)]
    vocab = {f"w{i}": i for i in range(1000)}
    with open(dataset_path, 'w') as f:
        for s in dataset: f.write(json.dumps(s) + '\n')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
else:
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

class SimpleMoE(nn.Module):
    def __init__(self, vocab_size=len(vocab)):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.fc = nn.Linear(64, vocab_size)
    def forward(self, x):
        return self.fc(self.embed(x).mean(dim=1))

model = SimpleMoE()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop with persistence
print("🚀 Starting training...")
global_step = 0
for epoch in range(1):
    for i, sample in enumerate(dataset):
        input_ids = torch.tensor([vocab[w] for w in sample["sentence"]]).unsqueeze(0)
        target = torch.tensor([vocab[sample["next_word"]]])
        
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        global_step += 1
        
        # Auto-checkpoint every 500 steps
        if global_step % 500 == 0:
            persistence.save_checkpoint(model.state_dict(), global_step, loss.item())
            print(f"💾 Checkpoint saved at step {global_step}")
