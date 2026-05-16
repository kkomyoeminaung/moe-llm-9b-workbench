# backend/self_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import json
import random
import time
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class SelfLearningSystem:
    """Auto self-learning and self-correction for MoE LLM"""
    
    def __init__(self, model, continuous_learner, rag_engine, vocab, shared_lock):
        self.model = model
        self.learner = continuous_learner
        self.rag = rag_engine
        self.vocab = vocab
        self._model_lock = shared_lock
        
        # Load persistence if available
        from backend.persistence_auto import get_persistence
        self.persistence = get_persistence()
        
        # Learning buffers
        self.interaction_history = deque(maxlen=self.persistence.load_setting("interaction_history_maxlen", 1000))
        saved_history = self.persistence.load_metadata("interaction_history", [])
        for item in saved_history:
            self.interaction_history.append(item)
            
        self.mistake_buffer = deque(maxlen=self.persistence.load_setting("mistake_buffer_maxlen", 200))
        saved_mistakes = self.persistence.load_metadata("mistake_buffer", [])
        for item in saved_mistakes:
            self.mistake_buffer.append(item)
            
        self.confidence_threshold = self.persistence.load_setting("confidence_threshold", 0.4)
        self.learning_interval = self.persistence.load_setting("learning_interval", 50)
        self.interaction_count = 0
        
        # Self-correction settings
        self.correction_enabled = True
        self.auto_learn_enabled = True
        self.is_learning = False
        
        # Performance tracking
        self.performance_stats = self.persistence.load_metadata("self_learning_stats", {
            "total_interactions": 0,
            "corrections_made": 0,
            "self_learn_sessions": 0,
            "avg_confidence": 0.0,
            "expert_performance": {str(i): {"correct": 0, "total": 0} for i in range(10)}
        })
        # Convert keys back to int if they were stringified in JSON
        if "expert_performance" in self.performance_stats:
            self.performance_stats["expert_performance"] = {
                int(k): v for k, v in self.performance_stats["expert_performance"].items()
            }
        
        # Background learning thread
        self.learning_thread = None

    def save_state(self):
        """Persist current state to disk"""
        self.persistence.save_metadata("interaction_history", list(self.interaction_history))
        self.persistence.save_metadata("mistake_buffer", list(self.mistake_buffer))
        self.persistence.save_metadata("self_learning_stats", self.performance_stats)
        self.persistence.save_setting("confidence_threshold", self.confidence_threshold)
        self.persistence.save_setting("learning_interval", self.learning_interval)
        print("💾 SelfLearningSystem state persisted.")

    def record_interaction(self, input_words: List[str], output_word: str, 
                          expert_id: int, confidence: float, was_correct: bool = None):
        """Record user interaction for self-learning"""
        self.interaction_history.append({
            "timestamp": time.time(),
            "input": input_words,
            "output": output_word,
            "expert": expert_id,
            "confidence": confidence,
            "was_correct": was_correct,
            "user_feedback": None  # Can be set later
        })
        
        self.interaction_count += 1
        self.performance_stats["total_interactions"] += 1
        
        # Update running average confidence
        total = self.performance_stats["avg_confidence"] * (self.performance_stats["total_interactions"] - 1)
        self.performance_stats["avg_confidence"] = (total + confidence) / self.performance_stats["total_interactions"]
        
        # Update expert performance
        if was_correct is not None:
            if was_correct:
                self.performance_stats["expert_performance"][expert_id]["correct"] += 1
            self.performance_stats["expert_performance"][expert_id]["total"] += 1
        
        # Check if correction needed
        if confidence < self.confidence_threshold:
            self.mistake_buffer.append({
                "input": input_words,
                "output": output_word,
                "expert": expert_id,
                "confidence": confidence,
                "timestamp": time.time()
            })
            
        # Trigger self-learning periodically
        if self.auto_learn_enabled and self.interaction_count >= self.learning_interval:
            self.trigger_self_learning()
            self.interaction_count = 0
            
    def provide_feedback(self, input_words: List[str], correct_output: str, expert_id: int = None):
        """User provides explicit feedback for correction"""
        # Find the interaction
        for interaction in self.interaction_history:
            if interaction["input"] == input_words:
                interaction["user_feedback"] = correct_output
                interaction["was_correct"] = False
                break
                
        # Add to mistake buffer for immediate correction
        self.mistake_buffer.append({
            "input": input_words,
            "output": correct_output,
            "expert": expert_id,
            "confidence": 0.0,
            "is_feedback": True
        })
        
        # Immediate correction
        self._correct_mistake(self.mistake_buffer[-1])
        
    def trigger_self_learning(self):
        """Trigger self-learning from accumulated mistakes"""
        if self.is_learning or len(self.mistake_buffer) == 0:
            return
            
        if self.learning_thread and self.learning_thread.is_alive():
            return
            
        self.learning_thread = threading.Thread(target=self._self_learn_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
    def _self_learn_loop(self):
        """Background self-learning loop"""
        self.is_learning = True
        print("🧠 Starting self-learning session...")
        
        # Learn from mistakes
        mistakes_to_learn = list(self.mistake_buffer)[-50:]  # Last 50 mistakes
        corrections_made = 0
        
        for mistake in mistakes_to_learn:
            if self._correct_mistake(mistake):
                corrections_made += 1
                
        # Generate synthetic examples for weak experts
        weak_experts = self._identify_weak_experts()
        for expert_id in weak_experts:
            self._generate_synthetic_examples(expert_id, count=10)
            
        self.performance_stats["self_learn_sessions"] += 1
        self.performance_stats["corrections_made"] += corrections_made
        
        # Save state after session
        self.save_state()
        
        print(f"✅ Self-learning complete. Corrected {corrections_made} mistakes.")
        self.is_learning = False
        
    def _correct_mistake(self, mistake: Dict) -> bool:
        """Correct a single mistake and learn from it"""
        try:
            input_words = mistake["input"]
            target_output = mistake["output"]
            expert_id = mistake.get("expert")
            
            # If expert not specified, detect from input
            if expert_id is None:
                expert_id = self._detect_expert_for_input(input_words)
                
            # Prepare training data
            input_ids = self._words_to_ids(input_words)
            target_id = self._word_to_id(target_output)
            
            if input_ids is None or target_id is None:
                return False
                
            loss_value = 0.0
            # Perform correction learning
            if self.model and hasattr(self.model, 'experts'):
                with self._model_lock:
                    if not hasattr(self, '_correction_optimizer'):
                        import torch.optim as optim
                        self._correction_optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

                    self.model.train()
                    self._correction_optimizer.zero_grad()
                    
                    # Forward pass
                    outputs, _ = self.model(input_ids)
                    
                    # Target preparation with correct device
                    device = outputs.device
                    target_tensor = torch.tensor([target_id], device=device)
                    
                    loss = F.cross_entropy(outputs, target_tensor)
                    
                    # Backward pass
                    loss.backward()
                    import torch.nn.utils as nn_utils
                    nn_utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self._correction_optimizer.step()
                    self.model.eval()
                    loss_value = loss.item()
                
            # Store in continuous learning memory
            if self.learner:
                self.learner.store_episode(input_words, target_output, expert_id, loss_value)
                
            return True
            
        except Exception as e:
            print(f"⚠️ Correction failed: {e}")
            return False
            
    def _identify_weak_experts(self) -> List[int]:
        """Identify experts that need improvement"""
        weak_experts = []
        for expert_id, stats in self.performance_stats["expert_performance"].items():
            if stats["total"] > 10:  # Enough samples
                accuracy = stats["correct"] / stats["total"]
                if accuracy < 0.6:  # Below 60% accuracy
                    weak_experts.append(expert_id)
        return weak_experts
        
    def _generate_synthetic_examples(self, expert_id: int, count: int):
        """Generate synthetic training examples for weak expert"""
        # Domain-specific patterns for each expert
        domain_patterns = {
            0: ["hello how are you", "good morning", "nice to meet you"],
            1: ["engine needs maintenance", "calculate force", "mechanical system"],
            2: ["the atom contains electrons", "energy cannot be created", "chemical reaction"],
            3: ["patient has fever", "take this medication", "symptoms include pain"],
            4: ["function returns value", "loop through array", "handle exception"],
            5: ["prayer is important", "faith gives hope", "sacred text says"],
            6: ["ancient civilization", "war changed history", "the king ruled"],
            7: ["market prices fluctuate", "supply and demand", "economic growth"],
            8: ["vote in election", "government policy", "citizen rights"],
            9: ["the novel begins", "poem has rhythm", "character development"]
        }
        
        patterns = domain_patterns.get(expert_id, domain_patterns[0])
        import re
        
        for i in range(count):
            pattern = random.choice(patterns)
            # Use Unicode aware split
            words = re.findall(r'[\w\u1000-\u109F]+', pattern.lower(), flags=re.UNICODE)
            if len(words) > 1:
                # Create simple next-word prediction example
                input_words = words[:-1]
                target_word = words[-1]
                
                self.mistake_buffer.append({
                    "input": input_words,
                    "output": target_word,
                    "expert": expert_id,
                    "confidence": 0.0,
                    "is_synthetic": True
                })
                
    def _detect_expert_for_input(self, words: List[str]) -> int:
        """Detect which expert should handle this input"""
        # Simple keyword-based detection
        domain_keywords = {
            0: ["hello", "hi", "how", "what", "who", "where", "when", "why"],
            1: ["engine", "mechanical", "circuit", "electrical", "motor", "force", "load"],
            2: ["science", "physics", "chemistry", "biology", "atom", "molecule", "energy"],
            3: ["medical", "patient", "doctor", "hospital", "disease", "symptom", "pain"],
            4: ["code", "program", "software", "function", "class", "variable", "algorithm"],
            5: ["god", "prayer", "faith", "church", "religion", "holy", "sacred"],
            6: ["history", "century", "ancient", "medieval", "war", "king", "empire"],
            7: ["economy", "market", "price", "money", "trade", "business", "finance"],
            8: ["politics", "government", "vote", "election", "policy", "law", "rights"],
            9: ["literature", "poem", "novel", "writer", "story", "book", "chapter"]
        }
        
        scores = {i: 0 for i in range(10)}
        for word in words:
            word_lower = word.lower()
            for domain, keywords in domain_keywords.items():
                if word_lower in keywords:
                    scores[domain] += 1
                    
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 0
        
    def _get_word_to_idx(self):
        if not hasattr(self, '_cached_word_to_idx'):
            if self.vocab:
                self._cached_word_to_idx = {v: k for k, v in self.vocab.items()}
            else:
                self._cached_word_to_idx = {}
        return self._cached_word_to_idx

    def _words_to_ids(self, words: List[str]) -> Optional[torch.Tensor]:
        """Convert words to token IDs"""
        if not self.vocab:
            return None
            
        word_to_idx = self._get_word_to_idx()
        ids = [word_to_idx.get(w, 0) for w in words[:64]]
            
        return torch.tensor([ids]).long() if ids else None
        
    def _word_to_id(self, word: str) -> Optional[int]:
        """Convert single word to ID"""
        if not self.vocab:
            return None
        return self._get_word_to_idx().get(word, 0)
        
    def get_status(self) -> Dict:
        """Get self-learning system status"""
        return {
            "enabled": self.auto_learn_enabled,
            "correction_enabled": self.correction_enabled,
            "total_interactions": self.performance_stats["total_interactions"],
            "corrections_made": self.performance_stats["corrections_made"],
            "self_learn_sessions": self.performance_stats["self_learn_sessions"],
            "avg_confidence": round(self.performance_stats["avg_confidence"], 3),
            "mistake_buffer_size": len(self.mistake_buffer),
            "interaction_history_size": len(self.interaction_history),
            "is_learning": self.is_learning,
            "expert_performance": {
                expert_id: {
                    "accuracy": round(stats["correct"] / stats["total"], 3) if stats["total"] > 0 else 0,
                    "samples": stats["total"]
                }
                for expert_id, stats in self.performance_stats["expert_performance"].items()
            }
        }
        
    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_interactions": 0,
            "corrections_made": 0,
            "self_learn_sessions": 0,
            "avg_confidence": 0.0,
            "expert_performance": {i: {"correct": 0, "total": 0} for i in range(10)}
        }
        self.mistake_buffer.clear()
        self.interaction_history.clear()
        
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for auto-correction"""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        
    def set_learning_interval(self, interval: int):
        """Set how many interactions between self-learning sessions"""
        self.learning_interval = max(10, interval)
