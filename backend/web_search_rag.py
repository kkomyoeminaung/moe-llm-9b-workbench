# backend/web_search_rag.py
"""Complete web search RAG with DuckDuckGo, caching, and auto-persistence"""

import aiohttp
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re
from backend.persistence_auto import get_persistence

class WebSearchRAG:
    """
    Complete web search RAG
    """
    
    def __init__(self, rag_engine, cache_ttl: int = 86400):  # 24 hour cache
        self.rag = rag_engine
        self.cache_ttl = cache_ttl
        self.persistence = get_persistence()
        self.cache = {}
        self.session = None
        
        # Load cache from persistence
        self._load_cache()
        
        print("✅ WebSearchRAG initialized")
    
    def _load_cache(self):
        """Load search cache from persistence"""
        cached = self.persistence.load_setting("web_search_cache", {})
        self.cache = cached
        print(f"📥 Loaded {len(self.cache)} cached searches")
    
    def _save_cache(self):
        """Save search cache to persistence"""
        # Clean old entries
        now = time.time()
        self.cache = {
            k: v for k, v in self.cache.items()
            if now - v.get("timestamp", 0) < self.cache_ttl
        }
        self.persistence.save_setting("web_search_cache", self.cache)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search the web"""
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached.get("timestamp", 0) < self.cache_ttl:
                return cached.get("results", [])
        
        results = []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=num_results)
                for r in ddgs_gen:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "content": r.get("body", ""),
                        "source": "web"
                    })
                        
        except Exception as e:
            print(f"⚠️ Web search error: {e}")
        
        # Cache results
        self.cache[cache_key] = {
            "results": results,
            "timestamp": time.time(),
            "query": query
        }
        self._save_cache()
        
        return results
    
    async def retrieve_with_web(self, query_words: List[str], use_web: bool = True, k: int = 5) -> List[List[str]]:
        """Retrieve from both local RAG and web"""
        query = ' '.join(query_words)
        all_results = []
        
        # 1. Local RAG first
        local_chunks = self.rag.retrieve_local(query_words, k=min(k, 3))
        for chunk in local_chunks:
            if chunk and chunk not in all_results:
                all_results.append(chunk)

        return all_results[:k]
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
        }
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
