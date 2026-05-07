# feedback/execution_feedback.py
"""Execute code and provide feedback for multi-file projects"""

import subprocess
import tempfile
import os
import shutil
from typing import Dict, List, Optional
from pathlib import Path

class CodeExecutor:
    """Execute and validate multi-file projects""" 
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or tempfile.mkdtemp(prefix="moe_workspace_")
        Path(self.workspace_root).mkdir(parents=True, exist_ok=True)
        print(f"📁 Workspace initialized at: {self.workspace_root}")

    def setup_project(self, files: Dict[str, str]):
        """Write files to the workspace"""
        for rel_path, content in files.items():
            full_path = Path(self.workspace_root) / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
        print(f"📦 Project files written: {list(files.keys())}")

    def install_dependencies(self) -> Dict:
        """Install dependencies from requirements.txt if it exists"""
        req_file = Path(self.workspace_root) / "requirements.txt"
        if not req_file.exists():
            return {"success": True, "message": "No requirements.txt found."}

        try:
            # Using -m pip to ensure we use the same python interpreter
            # In a real isolated env, we'd use a venv
            proc = subprocess.run(
                ['pip', 'install', '-r', str(req_file)],
                capture_output=True, text=True, timeout=60
            )
            if proc.returncode == 0:
                return {"success": True, "output": proc.stdout}
            else:
                return {"success": False, "error": proc.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_main(self, main_file: str = "main.py") -> Dict:
        """Execute the main entry point and capture output"""
        entry_point = Path(self.workspace_root) / main_file
        if not entry_point.exists():
            return {"success": False, "error": f"Entry point {main_file} not found."}

        try:
            proc = subprocess.run(
                ['python', str(entry_point)],
                capture_output=True, text=True, timeout=10,
                cwd=self.workspace_root
            )
            if proc.returncode == 0:
                return {"success": True, "output": proc.stdout}
            else:
                return {"success": False, "error": proc.stderr, "exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Remove the workspace"""
        if os.path.exists(self.workspace_root):
            shutil.rmtree(self.workspace_root)
            print(f"🧹 Workspace cleaned up: {self.workspace_root}")

    def create_zip(self, output_filename: str) -> str:
        """Create a zip archive of the project"""
        zip_path = shutil.make_archive(output_filename, 'zip', self.workspace_root)
        return zip_path
