"""
FastAPI application entry point.
Run: uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import get_settings
from backend.api.routers import auth, data, ml

cfg = get_settings()

app = FastAPI(
    title="Global Inflation Insights API",
    description="Production ML backend — FRED + yfinance + World Bank + PyTorch",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(data.router)
app.include_router(ml.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
