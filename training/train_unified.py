# training/train_unified.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return inputs_padded, targets
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import json
from tqdm import tqdm
from pathlib import Path
import time
import argparse
from model_unified import SparseMoE_Unified
from config import DEVICE, BATCH_SIZE, VOCAB_SIZE, EMBED_DIM, NUM_EXPERTS, EXPERT_LAYERS, HIDDEN_DIM, CONTEXT_LEN, LEARNING_RATE, WARMUP_STEPS, TOTAL_STEPS

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
        words = s['sentence'][:self.max_len]
        input_ids = [self.word_to_idx.get(w, 0) for w in words]
        target_id = self.word_to_idx.get(s['next_word'], 0)
        return torch.tensor(input_ids), torch.tensor(target_id)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    print(f"📱 Training on: {DEVICE}")
    
    data_path = Path("data/train.jsonl")
    if not data_path.exists():
        if args.fast_mode:
            print("🚀 Fast mode: Generating minimal dataset...")
            from generate_large_dataset import generate
            generate(num_samples=100)
        else:
            print("⚠️ data/train.jsonl not found.")
            return

    # Build vocabulary
    all_words = set()
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            all_words.update(data['sentence'])
            all_words.add(data['next_word'])
    
    word_to_idx = {w: i for i, w in enumerate(sorted(all_words)[:VOCAB_SIZE])}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    # Save Vocab for Backend
    with open("data/vocab.json", "w") as f:
        json.dump(idx_to_word, f)
    with open("data/word_to_idx.json", "w") as f:
        json.dump(word_to_idx, f)
    print("📋 Vocabulary saved for backend.")
    full_dataset = WordDataset(data_path, word_to_idx)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model Setup
    model = SparseMoE_Unified(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS,
        max_len=CONTEXT_LEN, expert_layers=EXPERT_LAYERS, ff_dim=HIDDEN_DIM
    ).to_device_optimized()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_val_loss = float('inf')

    # --- AUTO RESUME ---
    resume_options = [
        Path("checkpoints/latest.pt"),
        Path("checkpoints/last.pt"),
        Path("checkpoints/best.pt"),
        Path("checkpoints/moe_model_complete.pt")
    ]
    resume_path = None
    for opt in resume_options:
        if opt.exists():
            resume_path = opt
            break
    
    if resume_path:
        print(f"🔄 Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Optimizer state restored")
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
        else:
            model.load_state_dict(checkpoint)
    
    num_epochs = 1 if args.fast_mode else args.epochs

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, (input_ids, targets) in enumerate(pbar):
            input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)
            
            # Unpack model output: tuple(logits, expert_indices)
            outputs, _ = model(input_ids)
            loss = criterion(outputs, targets) + 0.01 * model.get_load_balancing_loss()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'step': global_step})
            
            # Frequent Checkpointing (Every 500 steps)
            if global_step % 500 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'best_val_loss': best_val_loss
                }, "checkpoints/last.pt")
                print(f" 💾 Auto-saved checkpoint at step {global_step}")

            if args.fast_mode and step >= 10: break

        # Validation per epoch
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for v_in, v_tar in val_loader:
                v_in, v_tar = v_in.to(DEVICE), v_tar.to(DEVICE)
                v_out, _ = model(v_in)
                v_loss += criterion(v_out, v_tar).item()
        v_loss /= len(val_loader)
        print(f"✅ Epoch {epoch+1} Val Loss: {v_loss:.4f}")
        
        # KEY: Switch back to training mode
        model.train()
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'best_val_loss': best_val_loss
            }, "checkpoints/best.pt")
            print("⭐ New best model saved.")

    final_name = "moe_model_complete.pt" if args.fast_mode else "moe_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'best_val_loss': best_val_loss
    }, f"checkpoints/{final_name}")
    print(f"🏁 Training finished. Artifact: checkpoints/{final_name}")

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    train()
