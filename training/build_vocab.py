# training/build_vocab.py
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

def build_vocab(dataset_path='data/train.jsonl', vocab_size=50000):
    print("📖 Building vocabulary...")
    word_counter = Counter()
    with open(dataset_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            word_counter.update(data['sentence'])
            word_counter.update([data['next_word']])
    
    most_common = word_counter.most_common(vocab_size)
    idx_to_word = {i: word for i, (word, _) in enumerate(most_common)}
    word_to_idx = {word: i for i, word in idx_to_word.items()}
    
    Path("data").mkdir(exist_ok=True)
    with open("data/vocab.json", 'w') as f:
        json.dump(idx_to_word, f)
    with open("data/word_to_idx.json", 'w') as f:
        json.dump(word_to_idx, f)
    print(f"✅ Vocabulary saved. Size: {len(idx_to_word)}")

if __name__ == "__main__":
    build_vocab()
