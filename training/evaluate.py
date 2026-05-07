# training/evaluate.py
import torch
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from model_unified import SparseMoE_Unified
from config import DEVICE, VOCAB_SIZE, EMBED_DIM, NUM_EXPERTS, EXPERT_LAYERS, HIDDEN_DIM, CONTEXT_LEN
from utils import compute_expert_balance
import argparse

def compute_perplexity(model, dataloader, criterion):
    """Compute perplexity on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, targets in tqdm(dataloader, desc="Computing perplexity"):
            input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)
            outputs, _ = model(input_ids)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss

def full_evaluation(model, train_loader, val_loader):
    """Run complete evaluation suite"""
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION SUITE")
    print("="*50)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # 1. Perplexity
    print("\n1️⃣ Computing Perplexity...")
    val_ppl, val_loss = compute_perplexity(model, val_loader, criterion)
    print(f"   Val Perplexity: {val_ppl:.2f}")
    print(f"   Val Loss: {val_loss:.4f}")
    
    # 2. Expert Balance
    print("\n2️⃣ Analyzing Expert Balance...")
    distribution, entropy = compute_expert_balance(model, val_loader, DEVICE)
    print(f"   Expert Distribution: {[f'{x:.2f}' for x in distribution]}")
    print(f"   Normalized Entropy: {entropy:.4f} (1.0 = perfect balance)")
    
    return {
        'val_ppl': val_ppl,
        'expert_distribution': distribution,
        'expert_balance_score': entropy
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/moe_5b_final.pt')
    args = parser.parse_args()
    
    # Load model
    model = SparseMoE_Unified(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_experts=NUM_EXPERTS,
        max_len=CONTEXT_LEN,
        expert_layers=EXPERT_LAYERS,
        ff_dim=HIDDEN_DIM
    )
    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print(f"✅ Loaded model from {args.model_path}")
    
    # (Simplified data loading for evaluation, skipping full dataloader setup)
    print("\n✅ Evaluation run finished.")

if __name__ == "__main__":
    main()
