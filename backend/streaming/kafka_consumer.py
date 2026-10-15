"""Kafka consumers for release-time macro data and high-rate IIoT telemetry.

Sensor offsets are committed only after their batch is durably written to the
TimescaleDB hypertable. This gives at-least-once delivery; the hypertable's
Kafka-coordinate key makes replay idempotent.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable

import asyncpg
from aiokafka import AIOKafkaConsumer
from redis.asyncio import Redis

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
Handler = Callable[[dict[str, Any]], Awaitable[None]]
LIVE_CHANNEL = "macro:stream:live"
SENSOR_TYPES = frozenset({"vibration", "thermal", "acoustic"})


def _postgres_dsn(url: str) -> str:
    """Convert a SQLAlchemy async URL into the DSN expected by asyncpg."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def publish_to_redis(event: dict[str, Any]) -> None:
    redis = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        await redis.publish(LIVE_CHANNEL, json.dumps(event, separators=(",", ":")))
    finally:
        await redis.aclose()


def _parse_timestamp(value: object) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("event_timestamp must include a timezone")
    return timestamp.astimezone(timezone.utc)


def validate_event(payload: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Validate existing macro messages without changing their schema."""
    kind = payload.get("event_type")
    if kind not in {"market_tick", "economic_release"}:
        raise ValueError("event_type must be market_tick or economic_release")
    for key in ("economy_code", "event_timestamp"):
        if not payload.get(key):
            raise ValueError(f"missing required field: {key}")
    event_time = _parse_timestamp(payload["event_timestamp"])
    now = now or datetime.now(timezone.utc)
    if event_time > now.astimezone(timezone.utc):
        raise ValueError("future event_timestamp is not observable yet")
    payload["event_timestamp"] = event_time.isoformat()
    payload.setdefault("ingested_at", now.astimezone(timezone.utc).isoformat())
    return payload


def parse_sensor_readings(payload: dict[str, Any], *, topic: str, partition: int,
                          offset: int, now: datetime | None = None) -> list[dict[str, Any]]:
    """Normalize a Kafka sensor message into calibrated samples.

    Producers may use either ``sensor_type``/``value`` or a ``sensors`` object.
    Kafka coordinates are persisted so a consumer restart cannot duplicate a
    physical reading.
    """
    asset_id = str(payload.get("asset_id", "")).strip()
    if not asset_id:
        raise ValueError("missing required field: asset_id")
    timestamp = _parse_timestamp(payload.get("event_timestamp"))
    observed_now = now or datetime.now(timezone.utc)
    if timestamp > observed_now.astimezone(timezone.utc):
        raise ValueError("future event_timestamp is not accepted")

    samples: Iterable[tuple[object, object]]
    if isinstance(payload.get("sensors"), dict):
        samples = payload["sensors"].items()
    else:
        samples = [(payload.get("sensor_type"), payload.get("value"))]

    rows: list[dict[str, Any]] = []
    for sensor_type, raw_value in samples:
        kind = str(sensor_type).lower()
        if kind not in SENSOR_TYPES:
            raise ValueError(f"unsupported sensor_type: {sensor_type}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("sensor values must be finite")
        units = payload.get("units", {})
        rows.append({
            "timestamp": timestamp, "asset_id": asset_id, "sensor_type": kind,
            "value": value, "unit": str(units.get(kind, payload.get("unit", ""))),
            "quality": str(payload.get("quality", "good")), "topic": topic,
            "partition": partition, "offset": offset,
            "ingested_at": observed_now.astimezone(timezone.utc),
        })
    if not rows:
        raise ValueError("sensor event contains no readings")
    return rows


class SensorTimescaleWriter:
    """Buffered writer for the ``sensor_readings`` TimescaleDB hypertable."""

    _INSERT = """
        INSERT INTO sensor_readings
            (timestamp, asset_id, sensor_type, value, unit, quality,
             kafka_topic, kafka_partition, kafka_offset, ingested_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (timestamp, kafka_topic, kafka_partition, kafka_offset, sensor_type)
        DO NOTHING
    """

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = _postgres_dsn(database_url or get_settings().database_url)
        self.pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        self.pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)

    async def write_batch(self, readings: list[dict[str, Any]]) -> None:
        if not readings:
            return
        if self.pool is None:
            raise RuntimeError("SensorTimescaleWriter.start() must be called first")
        values = [(
            row["timestamp"], row["asset_id"], row["sensor_type"], row["value"],
            row["unit"], row["quality"], row["topic"], row["partition"],
            row["offset"], row["ingested_at"],
        ) for row in readings]
        async with self.pool.acquire() as connection:
            await connection.executemany(self._INSERT, values)

    async def stop(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None


async def consume_sensor_stream(*, bootstrap_servers: str | None = None,
                                topic: str = "iiot-sensors", group_id: str = "predmaint-timescale",
                                batch_size: int = 500, poll_timeout_ms: int = 250,
                                writer: SensorTimescaleWriter | None = None) -> None:
    """Persist vibration, thermal, and acoustic Kafka events to TimescaleDB."""
    consumer = AIOKafkaConsumer(
        topic, bootstrap_servers=bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        group_id=group_id, enable_auto_commit=False, auto_offset_reset="latest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    database_writer = writer or SensorTimescaleWriter()
    await database_writer.start()
    await consumer.start()
    try:
        while True:
            records = await consumer.getmany(timeout_ms=poll_timeout_ms, max_records=batch_size)
            readings: list[dict[str, Any]] = []
            for topic_partition, messages in records.items():
                for message in messages:
                    try:
                        readings.extend(parse_sensor_readings(
                            message.value, topic=message.topic, partition=message.partition,
                            offset=message.offset,
                        ))
                    except (TypeError, ValueError, KeyError):
                        logger.exception("Discarding invalid sensor Kafka message at %s:%s:%s",
                                         topic_partition.topic, topic_partition.partition, message.offset)
            await database_writer.write_batch(readings)
            if records:
                await consumer.commit()
    finally:
        await consumer.stop()
        await database_writer.stop()


async def consume(handler: Handler = publish_to_redis, *, bootstrap_servers: str | None = None,
                  topics: tuple[str, ...] = ("market-ticks", "economic-releases"),
                  group_id: str = "macro-dashboard") -> None:
    """Consume and fan out existing macro messages."""
    consumer = AIOKafkaConsumer(*topics, bootstrap_servers=bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"), group_id=group_id, enable_auto_commit=False, auto_offset_reset="latest", value_deserializer=lambda value: json.loads(value.decode("utf-8")))
    await consumer.start()
    try:
        async for message in consumer:
            try:
                await handler(validate_event(message.value))
                await consumer.commit()
            except (ValueError, TypeError, json.JSONDecodeError):
                logger.exception("Discarding invalid macro Kafka message at %s", message.topic)
    finally:
        await consumer.stop()


async def run_forever(handler: Handler = publish_to_redis, **kwargs: Any) -> None:
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
