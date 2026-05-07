# training/config.py
import torch
import psutil

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_device_info():
    print(f"🔧 Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"   CPU Cores: {torch.get_num_threads()}")
        print(f"   RAM Available: {psutil.virtual_memory().total / 1e9:.1f} GB")

# Model configuration (Optimized for Colab Free)
if DEVICE.type == 'cuda':
    EXPERT_LAYERS = 3
    HIDDEN_DIM = 512
    BATCH_SIZE = 16
else:
    EXPERT_LAYERS = 2
    HIDDEN_DIM = 256
    BATCH_SIZE = 4

# Constants (Scaled down to prevent OOM)
VOCAB_SIZE = 10000
EMBED_DIM = 128
CONTEXT_LEN = 128
NUM_EXPERTS = 10
DOMAINS = ["chat", "engineering", "science", "medicine", "software_dev",
           "religion", "history", "economy", "politics", "literature"]
NUM_ACTIVE = 1              # Top-1 gating

LEARNING_RATE = 1e-3
WARMUP_STEPS = 500
TOTAL_STEPS = 500000
