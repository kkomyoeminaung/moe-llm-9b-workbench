# backend/dream_mode.py
import asyncio
import threading
import time
import random
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F

# Wikipedia domains for 10 experts
DOMAIN_WIKI_URLS = {
    0: "https://en.wikipedia.org/wiki/Chatbot",  # Chat
    1: "https://en.wikipedia.org/wiki/Engineering",  # Engineering
    2: "https://en.wikipedia.org/wiki/Science",  # Science
    3: "https://en.wikipedia.org/wiki/Medicine",  # Medicine
    4: "https://en.wikipedia.org/wiki/Software_engineering",  # Software
    5: "https://en.wikipedia.org/wiki/Religion",  # Religion
    6: "https://en.wikipedia.org/wiki/History",  # History
    7: "https://en.wikipedia.org/wiki/Economics",  # Economy
    8: "https://en.wikipedia.org/wiki/Politics",  # Politics
    9: "https://en.wikipedia.org/wiki/Literature",  # Literature
}

EXPERT_NAMES = ["chat", "engineering", "science", "medicine", "software_dev", 
                "religion", "history", "economy", "politics", "literature"]

import sys
from pathlib import Path
# Add training directory to sys.path
sys.path.append(str(Path(__file__).parent.parent / "training"))
from config import VOCAB_SIZE

from backend.logger import setup_logger

logger = setup_logger(__name__)

class DreamMode:
    """Curriculum-based Wikipedia learning during idle time"""
    
    def __init__(self, model, continuous_learner, rag_engine, shared_lock):
        self.model = model
        self.learner = continuous_learner
        self.rag = rag_engine
        self.is_learning = False
        self.stop_requested = False
        self.current_domain = 0
        self.stage = 0  # Curriculum stage
        self.learning_thread = None
        self.idle_threshold = 60  # 1 minute idle
        self.last_activity = time.time()
        self.dream_enabled = True # Master switch
        self._model_lock = shared_lock
        self._ingested_chunks_lock = threading.Lock()
        self._ingested_chunks = set()
        
        # Background monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_idle)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Curriculum stages
        self.curriculum = [
            {"domains": [0], "name": "Chat Basics"},  # stage 0
            {"domains": [0, 4], "name": "Chat + Software"},  # stage 1
            {"domains": [0, 1, 2], "name": "Chat + Engineering + Science"},  # stage 2
            {"domains": [0, 1, 2, 3], "name": "+ Medicine"},  # stage 3
            {"domains": [0, 1, 2, 3, 4], "name": "+ Software"},  # stage 4
            {"domains": [0, 1, 2, 3, 4, 6], "name": "+ History"},  # stage 5
            {"domains": [0, 1, 2, 3, 4, 6, 9], "name": "+ Literature"},  # stage 6
            {"domains": [0, 1, 2, 3, 4, 6, 9, 7], "name": "+ Economy"},  # stage 7
            {"domains": [0, 1, 2, 3, 4, 6, 9, 7, 8], "name": "+ Politics"},  # stage 8
            {"domains": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "name": "All Domains"},  # stage 9
        ]
        
        # Learning progress
        self.progress = {i: 0 for i in range(10)}  # 0-100%
        self.knowledge_base = {i: [] for i in range(10)}
        
    def _monitor_idle(self):
        """Periodically check if we should start/stop dream mode"""
        while True:
            if self.dream_enabled:
                self.start_if_idle()
            time.sleep(10)

    def record_activity(self):
        """Record user activity to reset idle timer"""
        self.last_activity = time.time()
        if self.is_learning:
            self.stop()
            
    def start_if_idle(self):
        """Start dream mode if idle for threshold"""
        idle_time = time.time() - self.last_activity
        if idle_time >= self.idle_threshold and not self.is_learning and not self.stop_requested:
            self.start()
            
    def start(self):
        """Start dream mode learning"""
        if self.is_learning:
            return
        self.stop_requested = False
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learn_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        logger.info(f"🌙 Dream mode started - learning curriculum stage {self.stage}")
        
    def stop(self):
        """Stop dream mode learning"""
        self.stop_requested = True
        # Don't join thread if it's the current thread
        if self.learning_thread and self.learning_thread.is_alive() and threading.current_thread() != self.learning_thread:
            self.learning_thread.join(timeout=2)
        self.is_learning = False
        logger.info("🌙 Dream mode paused")
        
    def _learn_loop(self):
        """Main learning loop for dream mode"""
        while not self.stop_requested and self.stage < len(self.curriculum):
            current_domains = self.curriculum[self.stage]["domains"]
            
            for domain in current_domains:
                if self.stop_requested:
                    break
                    
                # Learn from Wikipedia for this domain
                self._learn_from_wikipedia(domain)
                
                # Update progress
                self.progress[domain] = min(100, self.progress[domain] + 5)
                
                # Check if stage complete
                avg_progress = sum(self.progress[d] for d in current_domains) / len(current_domains)
                if avg_progress >= 80:
                    self.stage += 1
                    logger.info(f"📚 Curriculum completed stage {self.stage-1}, moving to stage {self.stage}")
                    break
                    
                # Small delay between domains
                time.sleep(2)
                    
            # Delay between rounds
            time.sleep(5)
            
        self.is_learning = False
        logger.info(f"🌙 Dream mode finished. Completed {self.stage} curriculum stages")
        
    def _learn_from_wikipedia(self, domain: int):
        """Learn from Wikipedia for a specific domain"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = DOMAIN_WIKI_URLS[domain]
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs[:20]])
            
            # Split into chunks (Unicode aware)
            import re
            words = re.findall(r'[\w\u1000-\u109F]+', text.lower(), flags=re.UNICODE)
            words = words[:1000] # Limit size
            chunks = [words[i:i+64] for i in range(0, len(words), 64)]
            
            # Learn from each chunk
            for chunk in chunks:
                if self.stop_requested:
                    break
                    
                # Store in knowledge base
                self.knowledge_base[domain].append(chunk)
                
                # Update RAG if not already present
                chunk_str = " ".join(chunk).strip()
                
                with self._ingested_chunks_lock:
                    should_train = False
                    if chunk_str not in self._ingested_chunks:
                        self.rag.add_document(chunk, domain=domain)
                        self._ingested_chunks.add(chunk_str)
                        should_train = True
                    
                # Short training step (outside lock)
                if should_train:
                    self._training_step(chunk, domain)
                    
                    # Clean up GPU cache occasionally
                    if domain % 3 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"⚠️ Wikipedia learning error for domain {domain}: {e}")
            
    def _get_word_id(self, w: str) -> int:
        vocab_size = VOCAB_SIZE
        if not hasattr(self, '_word_to_idx'):
            import os, json, hashlib
            vocab_path = "data/word_to_idx.json"
            if os.path.exists(vocab_path):
                with open(vocab_path, "r") as f:
                    self._word_to_idx = json.load(f)
            else:
                self._word_to_idx = {}
        if w in self._word_to_idx:
            return self._word_to_idx[w]
        import hashlib
        return int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16) % vocab_size

    def _training_step(self, words: List[str], domain: int):
        """Perform a single training step"""
        import torch.nn.functional as F
        if self.model is None or len(words) < 2:
            return

        try:
            device = next(self.model.parameters()).device
            is_ext = getattr(self.model, "is_external", False)
            
            with self._model_lock:
                self.model.train()
                
                if is_ext:
                    text = " ".join(words)
                    tokenizer = self.model.adapter.tokenizer
                    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    inputs["labels"] = inputs["input_ids"].clone()
                    
                    outputs = self.model.adapter.model(**inputs)
                    loss = outputs.loss
                else:
                    # Prepare data
                    input_ids = [self._get_word_id(w) for w in words[:-1]]
                    target_id = self._get_word_id(words[-1])
                    word_ids = torch.tensor([input_ids[:128]]).long().to(device)
                    targets = torch.tensor([target_id]).long().to(device)
                    
                    outputs, _ = self.model(word_ids)
                    
                    # outputs can be [batch, seq_len, vocab_size]. We only need the last token prediction.
                    if outputs.dim() == 3:
                        logits = outputs[:, -1, :]
                    else:
                        logits = outputs
                    
                    # Use scalar target for CrossEntropy
                    loss = F.cross_entropy(logits, targets)
                
                loss.backward()
                
                # Shared optimizer or local one
                if not hasattr(self, '_dream_optimizer'):
                    import torch.optim as optim
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    if not trainable_params:
                        return
                    self._dream_optimizer = optim.AdamW(trainable_params, lr=1e-5)
                
                # Optional: norm clipping for stability during "dreaming"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self._dream_optimizer.step()
                self._dream_optimizer.zero_grad()
                self.model.eval()
                
                # Periodic cache clear
                if random.random() < 0.1 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"⚠️ Training step failed: {e}")
        
    def get_status(self) -> Dict:
        """Get dream mode status"""
        return {
            "is_active": self.is_learning,
            "current_stage": self.stage,
            "stage_name": self.curriculum[self.stage]["name"] if self.stage < len(self.curriculum) else "Completed",
            "progress": self.progress,
            "idle_time": time.time() - self.last_activity,
            "idle_threshold": self.idle_threshold,
            "domains_learned": [d for d in self.progress if self.progress[d] > 0]
        }
        
    def set_threshold(self, seconds: int):
        """Alias for set_idle_threshold"""
        self.set_idle_threshold(seconds)

    def set_idle_threshold(self, seconds: int):
        """Set idle threshold in seconds"""
        self.idle_threshold = seconds
        
    def manual_learn_domain(self, domain: int, text: str):
        """Manually trigger learning for a domain"""
        words = text.split()
        chunks = [words[i:i+64] for i in range(0, len(words), 64)]
        for chunk in chunks:
            self.knowledge_base[domain].append(chunk)
            self.rag.add_document(chunk, domain=domain)
        self.progress[domain] = min(100, self.progress[domain] + 10)
