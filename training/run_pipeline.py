# training/run_pipeline.py
import subprocess
import time
import sys

def run(cmd_str, desc):
    print(f"🚀 {desc}...")
    import shlex
    cmd = shlex.split(cmd_str)
    subprocess.run(cmd, check=True)
    print(f"✅ {desc} finished.")

run(f"{sys.executable} training/generate_large_dataset.py", "Generating dataset")
run(f"{sys.executable} training/build_vocab.py", "Building vocab")
run(f"{sys.executable} training/train_unified.py", "Training model")
run(f"{sys.executable} training/evaluate.py", "Evaluating")
