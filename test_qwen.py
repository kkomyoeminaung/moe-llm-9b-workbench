from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print("pad:", type(tokenizer.pad_token_id), tokenizer.pad_token_id)
print("eos:", type(tokenizer.eos_token_id), tokenizer.eos_token_id)
