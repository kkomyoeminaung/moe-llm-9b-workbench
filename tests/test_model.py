# tests/test_model.py
import unittest
import torch
from training.model_unified import SparseMoE_Unified

class TestModel(unittest.TestCase):
    def test_forward(self):
        model = SparseMoE_Unified(vocab_size=1000, embed_dim=128, num_experts=4)
        input_ids = torch.randint(0, 1000, (2, 32))
        logits, expert_ids = model(input_ids)
        self.assertEqual(logits.shape, (2, 1000))
        self.assertEqual(expert_ids.shape, (2,))
