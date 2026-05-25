"""Celery app — async task queue for heavy ML/Monte Carlo jobs."""
from celery import Celery
from backend.core.config import get_settings

cfg = get_settings()

celery_app = Celery(
    "inflation_worker",
    broker=cfg.redis_url,
    backend=cfg.redis_url,
    include=["backend.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
