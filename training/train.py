import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import json
import wandb
import os
from tqdm import tqdm
from pathlib import Path
from model import SparseMoE
from config import *

# Dataset
class WordDataset:
    def __init__(self, path, word_to_idx, max_len=128):
        self.samples = []
        with open(path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        # Truncate/pad to max_len
        words = s['sentence'][:self.max_len]
        input_ids = [self.word_to_idx.get(w, 0) for w in words]
        target_id = self.word_to_idx.get(s['next_word'], 0)
        domain = s['domain']
        return torch.tensor(input_ids), torch.tensor(target_id), torch.tensor(domain)

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, targets, domains = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    domains = torch.stack(domains)
    return inputs_padded, targets, domains

# Warmup scheduler
def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)

# Validation
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    for input_ids, targets, domains in val_loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        
        outputs, _ = model(input_ids)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
    
    model.train()
    return total_loss / len(val_loader), total_correct / total_samples

# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)
    print(f"✅ Checkpoint saved to {path}")

# Main training function
def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Build vocabulary
    print("📖 Building vocabulary...")
    all_words = set()
    with open("data/train.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            all_words.update(data['sentence'])
            all_words.add(data['next_word'])
    
    word_to_idx = {w: i for i, w in enumerate(sorted(all_words)[:VOCAB_SIZE])}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    # Dataset
    full_dataset = WordDataset("data/train.jsonl", word_to_idx)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Model
    model = SparseMoE(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS)
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Scheduler: Warmup + Cosine
    total_steps = TOTAL_STEPS
    warmup_steps = WARMUP_STEPS
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Wandb
    wandb.init(project="word-moe-llm", config={
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "num_experts": NUM_EXPERTS,
        "total_steps": total_steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    })
    
    # Training
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if exists
    checkpoint_path = "checkpoints/latest.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint['step']
        print(f"🔄 Resumed from step {global_step}")
    
    for epoch in range(5):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for input_ids, targets, domains in pbar:
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            # Forward
            outputs, expert_ids, router_probs = model(input_ids, return_router_probs=True)
            
            # Loss
            ce_loss = criterion(outputs, targets)
            balance_loss = model.get_load_balancing_loss()
            loss = ce_loss + 0.01 * balance_loss
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            wandb.log({
                'loss': loss.item(),
                'ce_loss': ce_loss.item(),
                'balance_loss': balance_loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'step': global_step,
                'expert_entropy': -torch.sum(router_probs.mean(dim=0) * torch.log(router_probs.mean(dim=0) + 1e-8)).item()
            })
            
            pbar.set_postfix(loss=loss.item(), balance=balance_loss.item())
            global_step += 1
            
            # Validation every 1000 steps
            if global_step % 1000 == 0:
                val_loss, val_acc = validate(model, val_loader, device)
                wandb.log({'val_loss': val_loss, 'val_accuracy': val_acc, 'val_step': global_step})
                print(f"\n📊 Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, global_step, val_loss, "checkpoints/best.pt")
            
            # Save checkpoint every 5000 steps
            if global_step % 5000 == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, loss.item(), f"checkpoints/step_{global_step}.pt")
                save_checkpoint(model, optimizer, scheduler, global_step, loss.item(), "checkpoints/latest.pt")
            
            if global_step >= total_steps:
                break
        
        if global_step >= total_steps:
            break
    
    # Final save
    torch.save(model.state_dict(), "models/moe_5b_final.pt")
    print("✅ Training complete!")
    wandb.finish()

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    train()
