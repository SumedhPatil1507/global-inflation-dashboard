Place the append-only `macro_features.parquet` offline source here. Each row must
include `economy_code`, `event_timestamp` (UTC public release time), `ingested_at`
(UTC ingestion time), `reference_period`, and the four nullable numeric feature
columns. Keep revisions as new rows with their actual publication timestamps.

The feature entity supports USA, CAN, MEX, BRA, ARG, GBR, DEU, FRA, ITA, ESP,
CHE, SWE, NOR, POL, TUR, RUS, ZAF, IND, CHN, JPN, KOR, AUS, NZL, IDN, THA,
VNM, PHL, SAU, and ISR; add new economy codes without changing the schema.
