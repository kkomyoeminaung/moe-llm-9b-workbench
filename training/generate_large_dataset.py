# training/generate_large_dataset.py
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import time

# Domain word banks (High-quality vocabulary)
DOMAIN_WORDS = {
    0: ["greeting", "conversation", "dialogue", "interaction", "expression", "linguistics", "communication", "response"],
    1: ["mechanical", "optimization", "efficiency", "structure", "thermodynamics", "aerodynamics", "kinematics", "stress"],
    2: ["molecular", "cellular", "chemical", "quantum", "biological", "synthesis", "evolution", "metabolism"],
    3: ["neurological", "cardiovascular", "physiological", "pathological", "therapeutic", "diagnostic", "pharmacological"],
    4: ["asynchronous", "compilation", "encapsulation", "polymorphism", "abstraction", "recursion", "debugging", "profiling"],
    5: ["spirituality", "theology", "ritual", "philosophical", "eschatology", "doctrine", "orthodoxy", "meditation"],
    6: ["geopolitical", "imperialism", "sovereignty", "renaissance", "archeology", "diplomacy", "hegemony", "civilization"],
    7: ["macroeconomic", "fiscal", "monetary", "econometrics", "commodities", "equities", "volatility", "arbitrage"],
    8: ["legislative", "jurisdiction", "constitution", "ideology", "parliamentary", "suffrage", "bipartisan", "bureaucracy"],
    9: ["narrative", "allegory", "symbolism", "protagonist", "perspective", "anthology", "literary", "composition"]
}

COMMON_WORDS = ["the", "a", "an", "and", "of", "to", "in", "for", "on", "with", "is", "was", "will", "be"]

def generate_sample(domain):
    # Higher quality templates mimicking technical literature
    TEMPLATES = [
        "the {d1} {v} crucial {d2} in {d3}",
        "advanced {d1} {v} to {d2} and {d3}",
        "a systematic {d1} study {v} {d2}",
        "integrating {d1} with {d2} {v} the {d3}",
        "modern {d1} {v} based on {d2} principles",
        "the evolution of {d1} {v} complex {d2}"
    ]
    template = random.choice(TEMPLATES)
    words_domain = DOMAIN_WORDS[domain]
    verbs = ["demonstrates", "facilitates", "establishes", "optimizes", "characterizes", "validates", "enhances"]
    
    sentence_str = template.format(
        d1=random.choice(words_domain),
        d2=random.choice(words_domain),
        d3=random.choice(words_domain),
        v=random.choice(verbs)
    )
    full_words = sentence_str.split()
    return {"sentence": full_words[:-1], "domain": domain, "next_word": full_words[-1]}

def generate(num_samples=None, size_per_domain=None, output='data/train.jsonl'):
    Path("data").mkdir(exist_ok=True)
    output_path = Path(output)
    
    if num_samples is not None:
        # If specific count requested, generate across domains
        with open(output_path, 'w') as f:
            for _ in tqdm(range(num_samples), desc="Generating Data"):
                f.write(json.dumps(generate_sample(random.randint(0, 9))) + '\n')
        print(f"✅ Generated {num_samples} samples at {output}")
        return

    # Normal mode with domain logic
    size = size_per_domain if size_per_domain else 50000
    
    # --- RESUME LOGIC ---
    existing_samples = 0
    if output_path.exists():
        with open(output_path, 'r') as f:
            for _ in f:
                existing_samples += 1
    
    start_domain = existing_samples // size
    skip_samples = existing_samples % size
    
    if existing_samples > 0:
        print(f"🔄 Resuming dataset generation from Domain {start_domain}, Sample {skip_samples} (Total: {existing_samples})")
    
    mode = 'a' if existing_samples > 0 else 'w'
    with open(output_path, mode) as f:
        for domain in range(start_domain, 10):
            start_idx = skip_samples if domain == start_domain else 0
            pbar = tqdm(range(start_idx, size), desc=f"Domain {domain}")
            for _ in pbar:
                f.write(json.dumps(generate_sample(domain)) + '\n')
            skip_samples = 0

    print(f"✅ Dataset generation complete at {output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size_per_domain', type=int, default=10000)
    parser.add_argument('--output', type=str, default='data/train.jsonl')
    args = parser.parse_args()
    generate(size_per_domain=args.size_per_domain, output=args.output)

if __name__ == "__main__":
    main()
