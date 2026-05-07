# backend/logger.py
import logging
import sys
import os

def setup_logger(name: str, level=None):
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level, logging.INFO)
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    
    return logger
