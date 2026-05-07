import torch
from peft import get_peft_model, LoraConfig

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

def export_to_gguf(model):
    # This usually requires llama.cpp convert.py script
    print("Use llama.cpp/convert.py to export your model to GGUF.")

if __name__ == "__main__":
    print("QLoRA and GGUF setup prepared.")
