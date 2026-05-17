from backend.dream_mode import DreamMode
from backend.knowledge_ingestion import KnowledgeIngestion
from backend.self_learning import SelfLearningSystem
from backend.database import MoEDatabase
from agent.code_agent import CodeAgent
from feedback.execution_feedback import CodeExecutor
from typing import Dict, Any, Optional
import os
import threading

class SystemOrchestrator:
    def __init__(self, model, vocab: Dict[int, str], rag, learner):
        self.rag = rag
        self.learner = learner
        self.vocab = vocab
        self.model = model
        
        # Stability Flags (Disabled per User Request)
        self.ENABLE_DREAM = False
        self.ENABLE_SELF_LEARNING = False
        
        # Shared training lock
        self.model_lock = threading.Lock()
        
        self.dream = DreamMode(model, self.learner, self.rag, self.model_lock)
        self.ingestion = KnowledgeIngestion(self.rag, self.learner, model)
        self.self_learning = SelfLearningSystem(model, self.learner, self.rag, self.vocab, self.model_lock)
        
        # New autonomous architect components
        self.executor = CodeExecutor()
        self.code_agent = CodeAgent(model, self.executor, self.rag)
        
    def record_chat_interaction(self, input_words, output_word, expert_id, confidence):
        self.learner.store_episode(input_words, output_word, expert_id, confidence)
        
        if self.ENABLE_DREAM:
            self.dream.record_activity()
            
        if self.ENABLE_SELF_LEARNING:
            self.self_learning.record_interaction(input_words, output_word, expert_id, confidence)

    def build_software(self, project_name: str, requirements: str) -> Dict:
        """Autonomous software construction workflow"""
        print(f"🏗️ Building software: {project_name}")
        
        # Execute agent workflow
        result = self.code_agent.run(requirements)
        
        if result["success"]:
            # Zip the project for export
            export_path = os.path.join("exports", project_name)
            os.makedirs("exports", exist_ok=True)
            zip_file = self.executor.create_zip(export_path)
            result["zip_url"] = f"/api/download/{os.path.basename(zip_file)}"
            
        return result
