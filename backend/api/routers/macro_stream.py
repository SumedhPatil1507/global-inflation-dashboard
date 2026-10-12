"""Redis Pub/Sub to browser WebSocket fan-out for live macro updates."""
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from redis.asyncio import Redis
from backend.core.config import get_settings

router = APIRouter(tags=["macro-stream"])
CHANNEL = "macro:stream:live"


@router.websocket("/api/macro/stream/live")
async def live_macro_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    redis = Redis.from_url(get_settings().redis_url, decode_responses=True)
    pubsub = redis.pubsub()
    try:
        await pubsub.subscribe(CHANNEL)
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("type") == "message":
                # Payloads include the latest inflation metrics/factor updates.
                await websocket.send_text(message["data"])
            else:
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(CHANNEL)
        await pubsub.aclose()
        await redis.aclose()
