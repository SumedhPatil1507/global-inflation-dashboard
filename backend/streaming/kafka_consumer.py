"""Async Kafka ingestion for market ticks and economic release events.

Messages are JSON objects with `event_type` (market_tick or economic_release),
`economy_code`, `event_timestamp` and `ingested_at`. Event timestamps represent
when the value was observable; a future-dated release is rejected.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from aiokafka import AIOKafkaConsumer
from redis.asyncio import Redis
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
Handler = Callable[[dict[str, Any]], Awaitable[None]]
LIVE_CHANNEL = "macro:stream:live"


async def publish_to_redis(event: dict[str, Any]) -> None:
    """Publish validated tick/release messages for WebSocket subscribers.

    Producers can include computed `inflation_metrics` and `dynamic_factors` in
    the event; the stream forwards them without inventing values.
    """
    redis = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        await redis.publish(LIVE_CHANNEL, json.dumps(event, separators=(",", ":")))
    finally:
        await redis.aclose()


def validate_event(payload: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    kind = payload.get("event_type")
    if kind not in {"market_tick", "economic_release"}:
        raise ValueError("event_type must be market_tick or economic_release")
    for key in ("economy_code", "event_timestamp"):
        if not payload.get(key):
            raise ValueError(f"missing required field: {key}")
    event_time = datetime.fromisoformat(str(payload["event_timestamp"]).replace("Z", "+00:00"))
    if event_time.tzinfo is None:
        raise ValueError("event_timestamp must include a timezone")
    now = now or datetime.now(timezone.utc)
    if event_time.astimezone(timezone.utc) > now.astimezone(timezone.utc):
        raise ValueError("future event_timestamp is not observable yet")
    payload["event_timestamp"] = event_time.astimezone(timezone.utc).isoformat()
    payload.setdefault("ingested_at", now.astimezone(timezone.utc).isoformat())
    return payload


async def consume(handler: Handler = publish_to_redis, *, bootstrap_servers: str | None = None,
                  topics: tuple[str, ...] = ("market-ticks", "economic-releases"),
                  group_id: str = "macro-dashboard") -> None:
    consumer = AIOKafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    await consumer.start()
    try:
        async for message in consumer:
            try:
                event = validate_event(message.value)
                await handler(event)
                await consumer.commit()
            except (ValueError, TypeError, json.JSONDecodeError):
                logger.exception("Discarding invalid macro Kafka message at %s", message.topic)
    finally:
        await consumer.stop()


async def run_forever(handler: Handler = publish_to_redis, **kwargs: Any) -> None:
    """Restart the consumer on transient broker failures with bounded backoff."""
    delay = 1
    while True:
        try:
            await consume(handler, **kwargs)
            delay = 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Kafka consumer stopped; retrying in %ss", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
