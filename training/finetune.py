# training/finetune.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from pathlib import Path
from model_unified import SparseMoE_Unified
from utils import freeze_except_expert
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, VOCAB_SIZE, EMBED_DIM, NUM_EXPERTS, EXPERT_LAYERS, HIDDEN_DIM, CONTEXT_LEN
import argparse

class FineTuneDataset:
    def __init__(self, path, word_to_idx, domain_filter=None):
        self.samples = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if domain_filter is None or data['domain'] == domain_filter:
                    self.samples.append(data)
        self.word_to_idx = word_to_idx
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        input_ids = [self.word_to_idx.get(w, 0) for w in s['sentence'][:128]]
        target = self.word_to_idx.get(s['next_word'], 0)
        return torch.tensor(input_ids), torch.tensor(target)

def finetune_expert(model, train_loader, expert_id, epochs=3):
    """Fine-tune a specific expert on domain data"""
    print(f"\n🎯 Fine-tuning Expert {expert_id}")
    print("="*40)
    
    # Freeze other experts
    freeze_except_expert(model, expert_id)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE / 10  # Lower learning rate for fine-tuning
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for input_ids, targets in pbar:
            input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)
            
            outputs, _, _ = model(input_ids, return_details=True)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        print(f"Epoch {epoch+1}: loss={train_loss / len(train_loader):.4f}")
    
    return model

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return inputs_padded, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_id', type=int, required=True, help='Expert ID to fine-tune (0-9)')
    args = parser.parse_args()
    
    # Load vocab
    with open('data/word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)
    
    # Create datasets
    full_dataset = FineTuneDataset('data/train.jsonl', word_to_idx, domain_filter=args.expert_id)
    if len(full_dataset) == 0:
        print(f"⚠️ No samples found for domain {args.expert_id}. Skipping fine-tuning.")
        return

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Load base model
    model = SparseMoE_Unified(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_experts=NUM_EXPERTS
    )
    ckpt_path = Path("models/moe_5b_final.pt")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model = model.to(DEVICE)

    # Fine-tune
    finetune_expert(model, train_loader, args.expert_id)
    
    # Save
    torch.save(model.state_dict(), f"models/expert_{args.expert_id}_finetuned.pt")
    print(f"\n✅ Fine-tuning complete. Saved to models/expert_{args.expert_id}_finetuned.pt")

if __name__ == "__main__":
    main()
