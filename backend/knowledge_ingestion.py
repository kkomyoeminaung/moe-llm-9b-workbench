# backend/knowledge_ingestion.py
import os
import io
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import json
import re

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyPDF2 not installed. PDF support disabled.")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx not installed. Word support disabled.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️ BeautifulSoup not installed. HTML support disabled.")


from backend.utils import tokenize

class KnowledgeIngestion:
    """Process uploaded files (ZIP, PDF, WORD, TXT, HTML) into RAG"""
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.html': 'html',
        '.htm': 'html',
        '.zip': 'zip',
        '.json': 'json'
    }
    
    def __init__(self, rag_engine, continuous_learner, model=None):
        self.rag = rag_engine
        self.learner = continuous_learner
        self.model = model
        self.ingestion_history = []
        
    def process_upload(self, file_content: bytes, filename: str, domain: int = None) -> Dict:
        """Process uploaded file and ingest knowledge"""
        ext = Path(filename).suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            return {"error": f"Unsupported file type: {ext}", "success": False}
        
        result = {
            "filename": filename,
            "success": True,
            "chunks": 0,
            "words": 0,
            "domains": []
        }
        
        try:
            file_type = self.SUPPORTED_EXTENSIONS[ext]
            
            if file_type == 'zip':
                chunks, words, domains = self._process_zip(file_content, domain)
            elif file_type == 'pdf':
                chunks, words = self._process_pdf(file_content, domain)
                domains = [domain] if domain is not None else []
            elif file_type == 'docx':
                chunks, words = self._process_docx(file_content, domain)
                domains = [domain] if domain is not None else []
            elif file_type == 'html':
                chunks, words = self._process_html(file_content, domain)
                domains = [domain] if domain is not None else []
            elif file_type == 'text' or file_type == 'json':
                chunks, words = self._process_text(file_content, domain)
                domains = [domain] if domain is not None else []
            else:
                result["success"] = False
                result["error"] = f"Unknown file type: {file_type}"
                return result
                
            result["chunks"] = chunks
            result["words"] = words
            result["domains"] = domains
            
            # Add to history
            self.ingestion_history.append(result)
            return result
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            return result
    
    def _process_zip(self, file_content: bytes, domain: int = None) -> tuple:
        """Process ZIP file containing multiple documents"""
        total_chunks = 0
        total_words = 0
        domains = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "upload.zip"
            with open(zip_path, 'wb') as f:
                f.write(file_content)
                
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for file_info in zf.infolist():
                    if file_info.filename.endswith('/'):
                        continue
                        
                    ext = Path(file_info.filename).suffix.lower()
                    if ext in self.SUPPORTED_EXTENSIONS:
                        content = zf.read(file_info.filename)
                        file_type = self.SUPPORTED_EXTENSIONS[ext]
                        
                        try:
                            if file_type == 'pdf':
                                chunks, words = self._process_pdf(content, domain)
                            elif file_type == 'docx':
                                chunks, words = self._process_docx(content, domain)
                            elif file_type == 'html':
                                chunks, words = self._process_html(content, domain)
                            else:
                                chunks, words = self._process_text(content, domain)
                                
                            total_chunks += chunks
                            total_words += words
                            if domain is None:
                                domains.append(file_info.filename)
                        except Exception as e:
                            print(f"Error processing {file_info.filename}: {e}")
                            
        return total_chunks, total_words, domains
    
    def _process_pdf(self, file_content: bytes, domain: int = None) -> tuple:
        """Extract text from PDF"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not installed")
            
        text = ""
        with io.BytesIO(file_content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() or ""
                
        return self._process_text_content(text, domain)
    
    def _process_docx(self, file_content: bytes, domain: int = None) -> tuple:
        """Extract text from Word document"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed")
            
        with io.BytesIO(file_content) as doc_file:
            doc = docx.Document(doc_file)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
        return self._process_text_content(text, domain)
    
    def _process_html(self, file_content: bytes, domain: int = None) -> tuple:
        """Extract text from HTML"""
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup not installed")
            
        html = file_content.decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        # Clean up whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        return self._process_text_content(text, domain)
    
    def _process_text(self, file_content: bytes, domain: int = None) -> tuple:
        """Process plain text file"""
        text = file_content.decode('utf-8', errors='ignore')
        return self._process_text_content(text, domain)
    
    def _process_text_content(self, text: str, domain: int = None) -> tuple:
        """Split text into chunks and add to RAG"""
        # Split into words (Unicode aware for Myanmar etc)
        words = tokenize(text)
        
        if not words:
            return 0, 0
            
        # Split into chunks of 64 words
        chunk_size = 64
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
        
        # Add each chunk to RAG
        for chunk in chunks:
            if chunk:
                chunk_domain = domain if domain is not None else self._detect_domain(chunk)
                if self.rag:
                    self.rag.add_document(chunk, domain=chunk_domain)
                
                # Optional: trigger continuous learning
                if self.learner:
                    if hasattr(self.learner, 'store_episode'):
                        self.learner.store_episode(chunk, 'ingested', chunk_domain, 1.0)
                    elif hasattr(self.learner, 'store_experience'):
                        self.learner.store_experience(chunk, 'ingested', chunk_domain, 0.0)
                    
        return len(chunks), len(words)
    
    def _detect_domain(self, words: List[str]) -> int:
        """Simple domain detection based on keywords"""
        domain_keywords = {
            0: ["hello", "hi", "how", "chat", "talk", "conversation"],
            1: ["engine", "mechanical", "circuit", "electrical", "motor"],
            2: ["science", "physics", "chemistry", "biology", "atom"],
            3: ["medical", "patient", "doctor", "hospital", "disease"],
            4: ["code", "program", "software", "function", "class"],
            5: ["god", "prayer", "faith", "church", "religion"],
            6: ["history", "century", "ancient", "medieval", "war"],
            7: ["economy", "market", "price", "money", "trade"],
            8: ["politics", "government", "vote", "election", "policy"],
            9: ["literature", "poem", "novel", "writer", "story"]
        }
        
        scores = {i: 0 for i in range(10)}
        for word in words[:100]:
            for domain, keywords in domain_keywords.items():
                if word in keywords:
                    scores[domain] += 1
                    
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 0
    
    def get_status(self) -> Dict:
        return {
            "history": self.ingestion_history[-10:],
            "total_files": len(self.ingestion_history),
            "pdf_available": PDF_AVAILABLE,
            "docx_available": DOCX_AVAILABLE,
            "html_available": BS4_AVAILABLE
        }
