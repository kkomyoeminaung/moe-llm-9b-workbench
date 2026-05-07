# training/utils.py
import torch
import numpy as np

def compute_expert_balance(model, dataloader, device):
    """Compute expert utilization balance"""
    model.eval()
    expert_counts = torch.zeros(model.num_experts)
    
    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            _, expert_ids, _ = model(input_ids, return_details=True)
            
            for i in range(model.num_experts):
                expert_counts[i] += (expert_ids == i).sum().item()
    
    total = expert_counts.sum()
    if total > 0:
        distribution = expert_counts / total
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-8)).item()
        max_entropy = np.log(model.num_experts)
        normalized_entropy = entropy / max_entropy
    else:
        distribution = torch.zeros(model.num_experts)
        normalized_entropy = 0
    
    return distribution.tolist(), normalized_entropy

def save_best_model(model, val_loss, best_val_loss, path="models/best_model.pt"):
    """Save model if val_loss improved"""
    if val_loss < best_val_loss:
        print(f"✨ Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving...")
        torch.save(model.state_dict(), path)
        return val_loss
    return best_val_loss

def freeze_except_expert(model, expert_id):
    """Freeze all experts except specified one"""
    for name, param in model.named_parameters():
        if f"experts.{expert_id}" in name or "router" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print(f"🔒 Frozen all except Expert {expert_id} and Router")

def log_metrics(step, metrics):
    """Simple logger for training metrics"""
    log_str = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    with open("training_log.txt", "a") as f:
        f.write(log_str + "\n")
    # print(log_str)
