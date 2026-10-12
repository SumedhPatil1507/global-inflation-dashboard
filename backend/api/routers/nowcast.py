"""Mixed frequency CPI nowcast endpoint."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from backend.core.security import require_analyst
from backend.models.midas_nowcast import MIDASNowcaster

router = APIRouter(prefix="/api/quant", tags=["quant", "nowcasting"])


class MIDASNowcastRequest(BaseModel):
    daily_observations: list[dict]
    monthly_cpi: dict[str, float] = Field(description="CPI values keyed by reference month")
    release_dates: dict[str, str] = Field(description="Public release timestamp for each CPI month")
    target_period: str
    as_of: str
    lookback_days: int = 60
    alpha: float = 1.0


@router.post("/midas-nowcast")
async def midas_nowcast(req: MIDASNowcastRequest, user: dict = Depends(require_analyst)):
    """Fit release-time-safe MIDAS weights and estimate a CPI reference month."""
    try:
        data = pd.DataFrame(req.daily_observations)
        if "date" not in data:
            raise ValueError("daily_observations rows must contain a date field")
        daily = data.set_index("date").apply(pd.to_numeric, errors="coerce")
        cpi = pd.Series(req.monthly_cpi, dtype=float)
        releases = pd.Series(req.release_dates, dtype="object")
        model = MIDASNowcaster(lookback_days=req.lookback_days, alpha=req.alpha).fit(daily, cpi, releases)
        result = model.nowcast(daily, target_period=req.target_period, as_of=req.as_of)
        return {
            "target_period": result.target_period, "as_of": result.as_of,
            "nowcast": result.nowcast, "predictors": result.predictors,
            "training_observations": result.training_observations,
            "estimated_decay": model.decay,
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
