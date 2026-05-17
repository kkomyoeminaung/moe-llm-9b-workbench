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
        self.ENABLE_RAG = False # Explicitly track RAG status if needed, though app handles it
        
        # Shared training lock
        self.model_lock = threading.Lock()
        
        # Services (Disabled to speed up initialization and focus on chat)
        self.dream = None
        self.ingestion = None
        self.self_learning = None
        
        # New autonomous architect components
        self.executor = CodeExecutor()
        self.code_agent = CodeAgent(model, self.executor, self.rag)
        
    def record_chat_interaction(self, input_words, output_word, expert_id, confidence):
        if self.learner:
            self.learner.store_episode(input_words, output_word, expert_id, confidence)
        
        # Always record activity to keep the idle timer updated
        if self.dream:
            self.dream.record_activity()
        
        if self.ENABLE_SELF_LEARNING and self.self_learning:
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
