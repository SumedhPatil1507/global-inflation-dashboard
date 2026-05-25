"""JWT auth endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from backend.core.security import (authenticate_user, create_access_token,
                                    get_current_user)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "role":         user["role"],
        "username":     user["username"],
    }


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {"username": user.get("sub"), "role": user.get("role")}


@router.post("/refresh")
async def refresh(user: dict = Depends(get_current_user)):
    """Issue a fresh token for an already-authenticated user."""
    token = create_access_token({"sub": user["sub"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}
