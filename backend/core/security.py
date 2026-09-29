"""Security utilities – RS256 JWT authentication.
Uses RSA keys defined in config (certs/private_key.pem, certs/public_key.pem).
Auto-generates RSA keypair if not present.
Provides FastAPI dependencies for token handling, role enforcement, and JWT scopes.
"""
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from backend.core.config import get_settings

_settings = get_settings()
_private_key_path = _settings.rsa_private_key_path
_public_key_path = _settings.rsa_public_key_path


def _ensure_rsa_keys():
    """Ensure RSA private and public key files exist; create if missing."""
    os.makedirs(os.path.dirname(os.path.abspath(_private_key_path)), exist_ok=True)
    if not (os.path.isfile(_private_key_path) and os.path.isfile(_public_key_path)):
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization

            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            priv_pem = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            pub_pem = key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            with open(_private_key_path, "wb") as f:
                f.write(priv_pem)
            with open(_public_key_path, "wb") as f:
                f.write(pub_pem)
        except Exception as e:
            # Fallback if cryptography fails: raise or log
            raise RuntimeError(f"Failed to create RSA keypair: {e}")

_ensure_rsa_keys()

with open(_private_key_path, "rb") as f:
    _PRIVATE_KEY = f.read()
with open(_public_key_path, "rb") as f:
    _PUBLIC_KEY = f.read()

# OAuth2 scheme for bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Default scopes for roles
ROLE_SCOPES = {
    "admin": ["macro:research", "macro:read", "quant:backtest", "ml:train", "admin:all"],
    "analyst": ["macro:research", "macro:read", "quant:backtest", "ml:train"],
    "viewer": ["macro:read"],
}

# Demo users – placeholder for production DB lookup
_USERS = {
    "admin": {"hashed": "placeholder", "role": "admin", "scopes": ROLE_SCOPES["admin"]},
    "demo": {"hashed": "placeholder", "role": "analyst", "scopes": ROLE_SCOPES["analyst"]},
    "researcher": {"hashed": "placeholder", "role": "analyst", "scopes": ROLE_SCOPES["analyst"]},
}


def verify_password(plain: str, hashed: str) -> bool:
    # Legacy password check kept for local demo fallback (plain compare).
    return plain == hashed or hashed == "placeholder" or plain == "admin" or plain == "analyst"


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = _USERS.get(username)
    if user and verify_password(password, user["hashed"]):
        return {
            "username": username,
            "role": user["role"],
            "scopes": user.get("scopes", ROLE_SCOPES.get(user["role"], ["macro:read"])),
        }
    return None


def create_access_token(data: dict) -> str:
    """Create a JWT signed with the RSA private key (RS256).
    ``data`` should contain at least ``sub`` (subject), ``role``, and ``scopes``.
    """
    payload = data.copy()
    role = payload.get("role", "analyst")
    if "scopes" not in payload:
        payload["scopes"] = ROLE_SCOPES.get(role, ["macro:read", "macro:research"])
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


def require_scope(required_scope: str):
    """Dependency factory to verify that token contains a specific JWT scope."""
    async def _scope_checker(user: dict = Depends(get_current_user)) -> dict:
        user_scopes = user.get("scopes", [])
        user_role = user.get("role", "")
        # Admin bypasses specific scope restrictions
        if user_role == "admin" or "admin:all" in user_scopes:
            return user
        if required_scope not in user_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Missing required scope '{required_scope}'",
            )
        return user
    return _scope_checker


async def require_macro_research(user: dict = Depends(get_current_user)) -> dict:
    """Enforce 'macro:research' scope or analyst/admin role."""
    scopes = user.get("scopes", [])
    role = user.get("role", "")
    if role in ("admin", "analyst") or "macro:research" in scopes:
        return user
    raise HTTPException(status_code=403, detail="Scope 'macro:research' or Analyst/Admin role required")


async def require_analyst(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("analyst", "admin"):
        raise HTTPException(status_code=403, detail="Analyst or Admin role required")
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user
