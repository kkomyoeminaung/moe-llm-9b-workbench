import torch
import os
import subprocess
from peft import get_peft_model, LoraConfig

def find_convert_script():
    """Dynamically search for llama.cpp conversion script in Kaggle/Colab."""
    search_paths = [
        os.getcwd(),
        "/kaggle/working/llama.cpp",
        "/content/llama.cpp"
    ]
    for sp in search_paths:
        if os.path.exists(sp):
            for root, dirs, files in os.walk(sp):
                if 'convert-hf-to-gguf.py' in files:
                    return os.path.join(root, 'convert-hf-to-gguf.py')
    return "convert-hf-to-gguf.py" # Fallback

def apply_qlora(model):
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["attention", "ffn.0", "ffn.3"], # Target attention and FFN linear layers
        lora_dropout=0.05, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    return model

def export_to_gguf(model_path, output_path):
    # This usually requires llama.cpp convert.py script
    script = find_convert_script()
    print(f"Using conversion script: {script}")
    
    cmd = [
        "python3", script,
        model_path,
        "--outfile", output_path,
        "--outtype", "q8_0" # Default to Q8 quantization
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully exported to {output_path}")
    except Exception as e:
        print(f"Failed to export: {e}")

if __name__ == "__main__":
    print("QLoRA and GGUF setup prepared.")
