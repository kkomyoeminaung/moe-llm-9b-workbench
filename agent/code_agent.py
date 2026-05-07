# agent/code_agent.py
"""Autonomous multi-file software architect agent"""

import json
from typing import Dict, List, Optional
from feedback.execution_feedback import CodeExecutor
from backend.utils import get_vocab, generate_text

class CodeAgent:
    """Autonomous Software Architect Agent"""
    
    def __init__(self, model, executor: CodeExecutor, rag=None):
        self.model = model
        self.executor = executor
        self.rag = rag
    
    def run(self, task: str) -> Dict:
        """Fully autonomous software creation flow"""
        print(f"🚀 Architecture Task: {task}")
        
        # 1. Planning
        plan = self._create_plan(task)
        print(f"📋 Plan generated: {plan['files_needed']}")
        
        # 2. Coding (Social of Experts/Expert Selection would happen here in a real MoE)
        project_files = self._generate_files(task, plan)
        
        # 3. Virtual Workspace Setup
        self.executor.setup_project(project_files)
        
        # 4. Verification (Dependency + Execution)
        print("⚙️ Installing dependencies...")
        dep_result = self.executor.install_dependencies()
        if not dep_result["success"]:
            return {"success": False, "stage": "dependencies", "error": dep_result["error"]}
            
        print("🏃 Running verification...")
        exec_result = self.executor.execute_main(plan.get("main_file", "main.py"))
        
        # 5. Self-Debugging Loop (up to 2 retries)
        retries = 0
        while not exec_result["success"] and retries < 2:
            print(f"🔧 Debugging attempt {retries + 1}...")
            project_files = self._debug_files(project_files, exec_result["error"])
            self.executor.setup_project(project_files)
            exec_result = self.executor.execute_main(plan.get("main_file", "main.py"))
            retries += 1
            
        if not exec_result["success"]:
             return {"success": False, "stage": "execution", "error": exec_result["error"], "files": list(project_files.keys())}
             
        # 6. Finalization
        print("✅ Project verified successfully.")
        return {
            "success": True, 
            "output": exec_result["output"],
            "files": list(project_files.keys()),
            "workspace": self.executor.workspace_root
        }

    def _create_plan(self, task: str) -> Dict:
        """Plan the project structure based on task keywords"""
        task_lower = task.lower()
        files_needed = ["main.py", "requirements.txt"]
        
        # Simple rule-based planning (MoE would do this better in production)
        if any(k in task_lower for k in ["api", "web", "server", "fastapi", "flask"]):
             files_needed = ["main.py", "routes.py", "models.py", "requirements.txt"]
        elif any(k in task_lower for k in ["class", "oop", "architecture"]):
             files_needed = ["main.py", "models.py", "utils.py", "requirements.txt"]
        elif any(k in task_lower for k in ["data", "csv", "json", "analytics"]):
             files_needed = ["main.py", "data_processor.py", "requirements.txt"]
        elif any(k in task_lower for k in ["auth", "security", "login"]):
             files_needed = ["main.py", "auth.py", "utils.py", "requirements.txt"]

        return {
            "files_needed": files_needed,
            "main_file": "main.py"
        }

    def _generate_files(self, task: str, plan: Dict) -> Dict[str, str]:
        """Generate content for each file using model context"""
        files = {}
        
        for filename in plan["files_needed"]:
            context = f"Task: {task}. File: {filename}. Generate python code."
            # In a real MoE agent, we'd call the specifically trained expert
            # Here we use the generic generation logic
            content = self._generate_with_model(context)
            if filename == "requirements.txt" and not content.strip():
                content = "requests\n"
            files[filename] = content
        
        return files

    def _generate_with_model(self, prompt: str) -> str:
        """Helper to generate text using the MoE model"""
        if not self.model:
            return "# Model unavailable. Generic placeholder."
            
        try:
            vocab = get_vocab()
            # Generate a reasonable amount of code
            result = generate_text(
                self.model, vocab, prompt.split(), 
                max_new_words=100, temperature=0.5, top_k=20
            )
            return result["text"]
        except Exception as e:
            return f"# Placeholder for {prompt}. Error: {e}"

    def _debug_files(self, current_files: Dict, error_msg: str) -> Dict:
        """Placeholder for self-correction logic"""
        # Real logic would use the error_msg to fix the specific file
        return current_files
