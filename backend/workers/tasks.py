"""
Celery tasks — heavy computation runs here asynchronously.
Frontend polls /api/tasks/{task_id} for results.
"""
from backend.workers.celery_app import celery_app
from backend.services import ml_service


@celery_app.task(bind=True, name="tasks.train_model")
def task_train_model(self, records: list, epochs: int = 40) -> dict:
    self.update_state(state="PROGRESS", meta={"step": "training"})
    return ml_service.train_model(records, epochs)


@celery_app.task(bind=True, name="tasks.lstm_forecast")
def task_lstm_forecast(self, records: list, country: str,
                       forecast_years: int = 5, epochs: int = 60) -> dict:
    self.update_state(state="PROGRESS", meta={"step": "forecasting"})
    return ml_service.run_lstm_forecast(records, country, forecast_years, epochs)


@celery_app.task(bind=True, name="tasks.monte_carlo")
def task_monte_carlo(self, base_return: float, volatility: float,
                     horizon: int = 10, simulations: int = 300) -> dict:
    self.update_state(state="PROGRESS", meta={"step": "simulating"})
    return ml_service.run_monte_carlo(base_return, volatility, horizon, simulations)
