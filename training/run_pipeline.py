# training/run_pipeline.py
import subprocess
import time

def run(cmd_str, desc):
    print(f"🚀 {desc}...")
    import shlex
    cmd = shlex.split(cmd_str)
    subprocess.run(cmd, check=True)
    print(f"✅ {desc} finished.")

run("python training/generate_large_dataset.py", "Generating dataset")
run("python training/build_vocab.py", "Building vocab")
run("python training/train_unified.py", "Training model")
run("python training/evaluate.py", "Evaluating")
