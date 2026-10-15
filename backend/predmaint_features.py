"""Feast point-in-time features for predictive-maintenance sensor windows.

The source is the Timescale continuous aggregate, whose timestamps identify the
end of a completed five-minute window. `created_timestamp` preserves ingestion
time so historical training joins cannot see a feature before it existed.
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float64
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource


asset_sensor = Entity(
    name="asset_sensor",
    join_keys=["asset_id", "sensor_type"],
    description="One physical sensing channel on an industrial asset.",
)

sensor_window_source = PostgreSQLSource(
    name="timescale_sensor_feature_windows",
    query="""
        SELECT event_timestamp, asset_id, sensor_type, rms, kurtosis,
               crest_factor, created_timestamp
        FROM sensor_feature_windows
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

sensor_condition_features = FeatureView(
    name="sensor_condition_features",
    entities=[asset_sensor],
    ttl=timedelta(days=30),
    schema=[
        Field(name="rms", dtype=Float64),
        Field(name="kurtosis", dtype=Float64),
        Field(name="crest_factor", dtype=Float64),
    ],
    source=sensor_window_source,
    online=True,
    tags={
        "domain": "predictive_maintenance",
        "pit_semantics": "aggregate creation time",
        "window": "5m",
    },
)
