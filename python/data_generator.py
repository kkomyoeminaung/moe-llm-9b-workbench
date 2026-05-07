import json
import random

def generate_dataset(num_sentences=500000):
    vocab = ["word" + str(i) for i in range(50000)]
    with open("dataset.jsonl", "w") as f:
        for _ in range(num_sentences):
            sentence = [random.choice(vocab) for _ in range(10)]
            domain = random.randint(0, 9)
            next_word = random.choice(vocab)
            f.write(json.dumps({"sentence": sentence, "domain": domain, "next_word": next_word}) + "\n")

if __name__ == "__main__":
    generate_dataset()
    print("Generated 500k sentences.")
