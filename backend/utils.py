# backend/utils.py
import json
import hashlib
import torch
from pathlib import Path
import sys

# Ensure training config is reachable
sys.path.append(str(Path(__file__).parent.parent / "training"))
from config import VOCAB_SIZE, DEVICE, DOMAINS

_vocab = None
_word_to_idx = None

def get_vocab():
    global _vocab
    if _vocab is None:
        _vocab = {str(i): f"word_{i}" for i in range(VOCAB_SIZE)}
        vocab_path = Path("data/vocab.json")
        if vocab_path.exists():
            with open(vocab_path, "r") as f:
                _vocab = json.load(f)
    return _vocab

def get_word_to_idx():
    global _word_to_idx
    if _word_to_idx is None:
        vocab_path = Path("data/word_to_idx.json")
        if vocab_path.exists():
            with open(vocab_path, "r") as f:
                _word_to_idx = json.load(f)
        else:
            _word_to_idx = {}
    return _word_to_idx

def get_word_id(w: str) -> int:
    w2i = get_word_to_idx()
    if w in w2i:
        return w2i[w]
    import hashlib
    return int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16) % VOCAB_SIZE

def generate_text(model, vocab, initial_text_words, max_new_words=30, temperature=0.7, top_k=50, context_len=64):
    """Common autoregressive generation logic with metadata"""
    model.eval()
    generated_words = []
    total_confidence = 0
    expert_ids_used = []
    
    current_ids = torch.tensor([[get_word_id(w) for w in initial_text_words[-context_len:]]]).long().to(DEVICE)
    
    with torch.no_grad():
        for i in range(max_new_words):
            outputs, expert_id = model(current_ids)
            expert_ids_used.append(expert_id.item())
            
            # Take last logits if model returns sequence logits
            logits = outputs[0, -1, :] if outputs.dim() == 3 else outputs[0]
            probs = torch.softmax(logits / temperature, dim=-1)
            
            p, indices = torch.topk(probs, min(top_k, VOCAB_SIZE))
            p = p / p.sum()
            sampled_idx = torch.multinomial(p, 1).item()
            predicted_id = indices[sampled_idx].item()
            
            confidence = p[sampled_idx].item()
            total_confidence += confidence
            
            response_word = vocab.get(str(predicted_id), "unknown")
            generated_words.append(response_word)
            
            if response_word in [".", "!", "?", "<eos>", "unknown"] and i > 5:
                break
                
            new_id = torch.tensor([[predicted_id]]).long().to(DEVICE)
            current_ids = torch.cat([current_ids, new_id], dim=1)[:, -context_len:]
            
    final_response = " ".join(generated_words)
    avg_confidence = total_confidence / len(generated_words) if generated_words else 0
    main_expert_id = max(set(expert_ids_used), key=expert_ids_used.count) if expert_ids_used else 0
    
    return {
        "text": final_response,
        "avg_confidence": avg_confidence,
        "main_expert_id": main_expert_id,
        "expert_ids": expert_ids_used
    }
