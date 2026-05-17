import torch
import torch.nn.functional as F
from model import SparseMoE
from continuous_learning import EpisodicMemory
import json
import os
from torch.utils.data import Dataset, DataLoader

class MoEDataset(Dataset):
    def __init__(self, file_path, context_len=1024):
        self.data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except: pass
        self.context_len = context_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item.get("tokens", [0])
        # Simple padding/truncation
        if len(tokens) > self.context_len:
            tokens = tokens[:self.context_len]
        else:
            tokens = tokens + [0] * (self.context_len - len(tokens))
        return torch.tensor(tokens).long()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseMoE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    memory = EpisodicMemory()
    
    # Load dataset
    dataset = MoEDataset("data/dataset.jsonl")
    if len(dataset) == 0:
        print("Dataset empty. Generating dummy data for structure verification.")
        # Create dummy data if none exists
        os.makedirs("data", exist_ok=True)
        with open("data/dataset.jsonl", "w") as f:
            for _ in range(100):
                f.write(json.dumps({"tokens": torch.randint(0, 1000, (512,)).tolist()}) + "\n")
        dataset = MoEDataset("data/dataset.jsonl")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Training started on {device}...")
    for epoch in range(5):
        total_loss = 0
        for batch_idx, x in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            
            # Target is the next token (shifted x)
            targets = x[:, 1:].contiguous()
            inputs = x[:, :-1].contiguous()
            
            outputs, lb_loss = model(inputs)
            
            # Reshape for cross entropy
            logits = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Main loss + Load balancing loss
            main_loss = F.cross_entropy(logits, targets)
            loss = main_loss + 0.1 * lb_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f} (LB: {lb_loss.item():.4f})")
                
                # Periodic memory replay (if data exists)
                if memory.size() > 10:
                    memo_batch = memory.sample(4)
                    # Train on memory...
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f"checkpoints/moe_model_epoch_{epoch}.pt")

    print("Training finished.")

if __name__ == "__main__":
    train()
