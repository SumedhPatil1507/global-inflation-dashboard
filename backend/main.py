"""
FastAPI application entry point.
Run: uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import get_settings
from backend.api.routers import auth, data, ml, agent
from backend.services.backtester import router as quant_router

cfg = get_settings()

app = FastAPI(
    title="Global Inflation Insights API",
    description="Production Macro Research & ML backend — FRED + yfinance + World Bank + LangGraph Agent",
    version="2.1.0",
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
app.include_router(quant_router)
app.include_router(agent.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "features": ["langgraph-macro-agent", "rag-worldbank-rbi", "quant-backtest", "rs256-jwt"],
    }
