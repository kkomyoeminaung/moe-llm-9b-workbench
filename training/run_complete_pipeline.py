# training/run_complete_pipeline.py
import subprocess
import os
from pathlib import Path
import sys

def run_step(cmd, desc):
    print(f"\n{'='*50}")
    print(f"🚀 Step: {desc}")
    print(f"{'='*50}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {desc} successfully completed.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} failed with error code {e.returncode}")
        sys.exit(1)

def main():
    # Ensure directories exist
    for d in ["data", "models", "checkpoints"]:
        Path(d).mkdir(exist_ok=True)

    # 1. Generate Dataset
    if not os.path.exists("data/train.jsonl"):
        run_step("python training/generate_large_dataset.py", "Generating dataset")
    else:
        print("⏭️ Dataset already exists, skipping generation.")

    # 2. Build Vocabulary
    if not os.path.exists("data/word_to_idx.json"):
        run_step("python training/build_vocab.py", "Building vocabulary")
    else:
        print("⏭️ Vocabulary already exists, skipping build.")

    # 3. Unified Training
    run_step("python training/train_unified.py", "Starting MoE Unified Training")

    # 4. Evaluate
    run_step("python training/evaluate.py", "Evaluating model")

    # 5. Export and Quantize (if applicable)
    # run_step("python training/03_quantize_export.ipynb", "Quantizing model")

    print("\n🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
