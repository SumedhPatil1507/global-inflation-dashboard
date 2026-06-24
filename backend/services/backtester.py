import numpy as np
import pandas as pd
from backend.services.data_warehouse import get_historical_matrix
from fastapi import APIRouter, Depends, HTTPException
from backend.core.security import require_analyst
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter(prefix="/api/quant", tags=["quant"])

class BacktestRequest(BaseModel):
    tickers: List[str]
    indicators: List[str]
    start_date: str  # ISO format e.g., "2020-01-01"
    end_date: str

class BacktestResult(BaseModel):
    equity_curve: List[float]
    dates: List[str]
    sharpe: float
    max_drawdown: float
    total_return: float
    annualized_volatility: float

def _calc_metrics(equity: pd.Series) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if not returns.empty else 0.0
    drawdown = (equity.cummax() - equity) / equity.cummax()
    max_dd = drawdown.max() if not drawdown.empty else 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0.0
    ann_vol = returns.std() * np.sqrt(252) if not returns.empty else 0.0
    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "annualized_volatility": ann_vol,
    }

@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(req: BacktestRequest, user: dict = Depends(require_analyst)):
    # Fetch historical matrix for given tickers/indicators
    matrix = await get_historical_matrix(req.tickers, req.indicators, req.start_date)
    if matrix.empty:
        raise HTTPException(status_code=400, detail="No data for given parameters")
    # Simple equity curve: sum of selected columns (placeholder logic)
    equity = matrix[req.tickers].sum(axis=1).fillna(0.0)
    metrics = _calc_metrics(equity)
    return BacktestResult(
        equity_curve=equity.tolist(),
        dates=[str(d) for d in equity.index],
        sharpe=metrics["sharpe"],
        max_drawdown=metrics["max_drawdown"],
        total_return=metrics["total_return"],
        annualized_volatility=metrics["annualized_volatility"],
    )
