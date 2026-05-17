from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('gpt2')
test_msgs = [{"role": "user", "content": "hello"}]
try:
    input_ids = tokenizer.apply_chat_template(test_msgs, return_tensors="pt")
    print("Success:", input_ids.shape)
except Exception as e:
    print("Error:", e)
