# tokenizer/code_tokenizer.py
"""Subword tokenizer for code - BPE based"""

import tokenizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import json
from pathlib import Path

class CodeBPETokenizer:
    """Byte-Pair Encoding tokenizer optimized for code"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        
        # Special tokens for code
        self.special_tokens = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            "<PRE>", "<SUF>", "<MID>",           # FIM tokens
            "<CURSOR>",                          # Cursor position
            "<FILE>", "<REPO>", "<IMPORT>",      # Repo context
            "<EXEC>", "<ERROR>", "<OUTPUT>",     # Execution feedback
            "<THOUGHT>", "<ACTION>", "<OBS>",    # Agent loop
        ]
        
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize BPE tokenizer"""
        # Use BPE model
        self.tokenizer = Tokenizer(models.BPE())
        
        # Pre-tokenizer: split on whitespace and punctuation
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Decoder
        self.tokenizer.decoder = decoders.ByteLevel()
        
        # Post-processor
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
    def train(self, files: list, vocab_size: int = 50000):
        """Train tokenizer on code corpus"""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True
        )
        
        self.tokenizer.train(files, trainer)
        
        # Save
        self.tokenizer.save("tokenizers/code_bpe.json")
        
        print(f"✅ Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> dict:
        """Encode text to token IDs"""
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "offsets": encoding.offsets
        }
    
    def decode(self, ids: list) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
    def apply_fim(self, prefix: str, suffix: str, middle: str = "") -> dict:
        """Apply Fill-in-the-Middle format"""
        # Format: <PRE> prefix <SUF> suffix <MID> middle
        fim_text = f"<PRE> {prefix} <SUF> {suffix} <MID> {middle}"
        return self.encode(fim_text)
    
    def add_cursor(self, text: str, cursor_pos: int) -> dict:
        """Add cursor position token"""
        before = text[:cursor_pos]
        after = text[cursor_pos:]
        cursor_text = f"{before}<CURSOR>{after}"
        return self.encode(cursor_text)

# Load or create tokenizer
def get_tokenizer(vocab_size: int = 50000) -> CodeBPETokenizer:
    tokenizer_path = Path("tokenizers/code_bpe.json")
    
    if tokenizer_path.exists():
        # Load existing
        tokenizer = CodeBPETokenizer(vocab_size)
        tokenizer.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"✅ Loaded existing tokenizer (vocab: {tokenizer.tokenizer.get_vocab_size()})")
    else:
        # Need to train
        print("⚠️ No tokenizer found. Please train first:")
        print("   python tokenizer/train_tokenizer.py --data /path/to/code/corpus")
        tokenizer = None
    
    return tokenizer
