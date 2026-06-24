# """Security utilities – RS256 JWT authentication.
# Uses RSA keys defined in config (certs/private_key.pem, certs/public_key.pem).
# Provides FastAPI dependencies for token handling and role enforcement.
#"""
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt, jwk
from backend.core.config import get_settings

# Load RSA keys once
_settings = get_settings()
_private_key_path = _settings.rsa_private_key_path
_public_key_path = _settings.rsa_public_key_path

if not (os.path.isfile(_private_key_path) and os.path.isfile(_public_key_path)):
    raise RuntimeError("RSA key files not found for JWT operations")

with open(_private_key_path, "rb") as f:
    _private_key = f.read()
with open(_public_key_path, "rb") as f:

# OAuth2 scheme for bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Demo users – placeholder for production DB lookup
_USERS = {
    "admin": {"hashed": "placeholder", "role": "admin"},
    "demo": {"hashed": "placeholder", "role": "analyst"},
}

def verify_password(plain: str, hashed: str) -> bool:
    # Legacy password check kept for local demo fallback (plain compare).
    return plain == hashed

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = _USERS.get(username)
    if user and verify_password(password, user["hashed"]):
        return {"username": username, "role": user["role"]}
    return None

def create_access_token(data: dict) -> str:
    """Create a JWT signed with the RSA private key.
    ``data`` should contain at least ``sub`` (subject) and ``role``.
    """
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=_settings.jwt_expire_min)
    token = jwt.encode(payload, _PRIVATE_KEY, algorithm=_settings.jwt_algorithm)
    return token

def decode_token(token: str) -> dict:
    """Decode and verify a JWT using the RSA public key."""
    try:
        return jwt.decode(token, _PUBLIC_KEY, algorithms=[_settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency that returns the JWT payload as a dict."""
    return decode_token(token)

async def require_analyst(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("analyst", "admin"):
        raise HTTPException(status_code=403, detail="Analyst or Admin role required")
    return user

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user
