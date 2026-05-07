# backend/auth.py
import time
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Simple token validation (e.g., "secret-token")
    if credentials.credentials != "secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"username": "user"}
