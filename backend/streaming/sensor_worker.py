"""Container entry point for the IIoT Kafka-to-TimescaleDB consumer."""
import asyncio

from backend.streaming.kafka_consumer import consume_sensor_stream


if __name__ == "__main__":
    asyncio.run(consume_sensor_stream())
