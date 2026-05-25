"""ML endpoints — submit jobs to Celery, poll for results."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.core.security import require_analyst
from backend.workers.tasks import (task_train_model, task_lstm_forecast,
                                    task_monte_carlo)
from celery.result import AsyncResult
from backend.workers.celery_app import celery_app

router = APIRouter(prefix="/api/ml", tags=["ml"])


class TrainRequest(BaseModel):
    records:     list[dict]
    epochs:      int = 40


class ForecastRequest(BaseModel):
    records:      list[dict]
    country:      str
    forecast_years: int = 5
    epochs:       int = 60


class MonteCarloRequest(BaseModel):
    base_return:  float
    volatility:   float
    horizon:      int = 10
    simulations:  int = 300


@router.post("/train")
async def submit_train(req: TrainRequest, user: dict = Depends(require_analyst)):
    task = task_train_model.delay(req.records, req.epochs)
    return {"task_id": task.id, "status": "queued"}


@router.post("/forecast")
async def submit_forecast(req: ForecastRequest, user: dict = Depends(require_analyst)):
    task = task_lstm_forecast.delay(req.records, req.country,
                                    req.forecast_years, req.epochs)
    return {"task_id": task.id, "status": "queued"}


@router.post("/monte-carlo")
async def submit_monte_carlo(req: MonteCarloRequest, user: dict = Depends(require_analyst)):
    task = task_monte_carlo.delay(req.base_return, req.volatility,
                                   req.horizon, req.simulations)
    return {"task_id": task.id, "status": "queued"}


@router.get("/tasks/{task_id}")
async def get_task_result(task_id: str, user: dict = Depends(require_analyst)):
    result = AsyncResult(task_id, app=celery_app)
    if result.state == "PENDING":
        return {"task_id": task_id, "status": "pending"}
    elif result.state == "PROGRESS":
        return {"task_id": task_id, "status": "running", "meta": result.info}
    elif result.state == "SUCCESS":
        return {"task_id": task_id, "status": "success", "result": result.result}
    elif result.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(result.info))
    return {"task_id": task_id, "status": result.state}
