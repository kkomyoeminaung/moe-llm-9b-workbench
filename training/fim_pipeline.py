# training/fim_pipeline.py
"""Fill-in-the-Middle training for code completion"""

import torch
import torch.nn as nn
import random
from typing import List, Tuple

class FIMDataset(torch.utils.data.Dataset):
    """Dataset for Fill-in-the-Middle training"""
    
    def __init__(self, code_files: List[str], tokenizer, fim_rate=0.5):
        self.code_files = code_files
        self.tokenizer = tokenizer
        self.fim_rate = fim_rate
        
        # Load code samples
        self.samples = []
        for path in code_files:
            try:
                with open(path, 'r') as f:
                    self.samples.append(f.read())
            except:
                pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        code = self.samples[idx]
        
        if random.random() < self.fim_rate:
            return self._create_fim_sample(code)
        else:
            return self._create_standard_sample(code)
    
    def _create_fim_sample(self, code: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create FIM sample: <PRE> prefix <SUF> suffix <MID> middle"""
        lines = code.split('\n')
        if len(lines) < 3:
            return self._create_standard_sample(code)
        
        split_point = random.randint(1, len(lines) - 2)
        prefix = '\n'.join(lines[:split_point])
        middle = '\n'.join(lines[split_point:split_point+1])
        suffix = '\n'.join(lines[split_point+1:])
        
        fim_text = f"<PRE> {prefix} <SUF> {suffix} <MID> {middle}"
        encoding = self.tokenizer.encode(fim_text)
        input_ids = torch.tensor(encoding["input_ids"])
        target_ids = input_ids.clone()
        return input_ids, target_ids
    
    def _create_standard_sample(self, code: str):
        """Standard next-token prediction sample"""
        encoding = self.tokenizer.encode(code)
        input_ids = torch.tensor(encoding["input_ids"])
        target_ids = input_ids.clone()
        return input_ids, target_ids
