"""Feast definitions for release-time-safe macro features.

`event_timestamp` is the instant a value became public (not the period it describes).
This keeps historical joins from exposing unreleased CPI/PPI/rate observations.
"""
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, String
from datetime import timedelta

economy = Entity(name="economy", join_keys=["economy_code"])

macro_source = FileSource(
    name="macro_release_source",
    path="data/macro_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="ingested_at",
)

macro_features = FeatureView(
    name="macro_features",
    entities=[economy],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="yield_spread_10y_2y", dtype=Float64),
        Field(name="cpi_yoy", dtype=Float64),
        Field(name="ppi_yoy", dtype=Float64),
        Field(name="policy_rate", dtype=Float64),
        Field(name="reference_period", dtype=String),
    ],
    source=macro_source,
    online=True,
    tags={"domain": "macro", "pit_semantics": "public_release_time"},
)
