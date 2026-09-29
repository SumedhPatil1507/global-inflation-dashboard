"""Macro Research Agent Endpoint with RS256 JWT Scoped Authorization."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from backend.core.security import require_macro_research, require_scope, get_current_user
from backend.services.macro_agent import run_macro_research

router = APIRouter(prefix="/api/macro", tags=["macro-agent"])


class ResearchQueryRequest(BaseModel):
    query: str = Field(..., description="Macro research question, e.g., 'Why did inflation move this quarter?'")
    country: Optional[str] = Field("IND", description="Target ISO country code (e.g., IND, USA, GBR, CHN)")
    start_year: Optional[int] = Field(2020, ge=2000, le=2026, description="Start year")
    end_year: Optional[int] = Field(2024, ge=2000, le=2026, description="End year")
    include_backtest: Optional[bool] = Field(True, description="Whether to include quantitative portfolio backtesting")
    backtest_tickers: Optional[List[str]] = Field(default_factory=lambda: ["GLD", "SPY", "TIP"])


class ResearchResponse(BaseModel):
    query: str
    country: str
    plan_steps: List[str]
    synthesis: str
    citations: List[Dict[str, Any]]
    time_series_data: Dict[str, Any]
    backtest_result: Optional[Dict[str, Any]] = None
    execution_trace: List[Dict[str, Any]]


@router.post("/research", response_model=ResearchResponse)
async def execute_macro_research(
    req: ResearchQueryRequest,
    user: dict = Depends(require_macro_research),
):
    """
    Executes LangGraph Macro Research Agent with:
    1. Multi-step reasoning and plan decomposition
    2. Real macro time-series retrieval via FRED / World Bank
    3. Quantitative portfolio backtesting via /api/quant/backtest
    4. Dense RAG retrieval over World Bank / RBI reports and macroeconomic news
    5. Cited synthesis and policy analysis

    Requires JWT token with 'macro:research' scope (or analyst/admin role).
    """
    try:
        result = run_macro_research(
            query=req.query,
            country=req.country or "IND",
            start_year=req.start_year or 2020,
            end_year=req.end_year or 2024,
            include_backtest=req.include_backtest,
            backtest_tickers=req.backtest_tickers,
        )
        return ResearchResponse(
            query=result.get("query", req.query),
            country=result.get("country", req.country),
            plan_steps=result.get("plan_steps", []),
            synthesis=result.get("synthesis", ""),
            citations=result.get("citations", []),
            time_series_data=result.get("time_series_data", {}),
            backtest_result=result.get("backtest_result"),
            execution_trace=result.get("execution_trace", []),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Macro research agent execution error: {str(e)}",
        )


@router.get("/rag/reports")
async def get_rag_reports(
    query: str = "inflation",
    country: Optional[str] = None,
    top_k: int = 4,
    user: dict = Depends(require_macro_research),
):
    """Retrieve raw chunked reports and excerpts from World Bank / RBI knowledge base."""
    from backend.services.macro_rag import retrieve_macro_reports
    results = retrieve_macro_reports(query=query, country=country, top_k=top_k)
    return {"query": query, "country": country, "results": results, "count": len(results)}
