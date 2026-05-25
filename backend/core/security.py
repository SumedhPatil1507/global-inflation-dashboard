"""JWT stateless auth — issue and verify tokens."""
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from backend.core.config import get_settings

pwd_ctx    = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2     = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Demo users — replace with Supabase DB lookup in production
_USERS = {
    "admin": {"hashed": pwd_ctx.hash("admin"),   "role": "admin"},
    "demo":  {"hashed": pwd_ctx.hash("demo123"), "role": "analyst"},
}


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = _USERS.get(username)
    if user and verify_password(password, user["hashed"]):
        return {"username": username, "role": user["role"]}
    return None


def create_access_token(data: dict) -> str:
    cfg     = get_settings()
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=cfg.jwt_expire_min)
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def decode_token(token: str) -> dict:
    cfg = get_settings()
    try:
        return jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token")


async def get_current_user(token: str = Depends(oauth2)) -> dict:
    return decode_token(token)


async def require_analyst(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("analyst", "admin"):
        raise HTTPException(status_code=403, detail="Analyst or Admin role required")
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user
