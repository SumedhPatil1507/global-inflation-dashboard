"""Point-in-time regression tests for macro feature retrieval."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

feast = pytest.importorskip("feast")
from feast import FileSource, FeatureStore


def test_get_historical_features_hides_unreleased_values(tmp_path):
    """A release is invisible before its public timestamp and visible afterward."""
    from backend.features import economy, macro_features

    data_path = tmp_path / "macro_features.parquet"
    pd.DataFrame([
        {
            "economy_code": "USA",
            "event_timestamp": pd.Timestamp("2024-02-15T13:30:00Z"),
            "ingested_at": pd.Timestamp("2024-02-15T13:30:02Z"),
            "reference_period": "2024-01",
            "yield_spread_10y_2y": -0.25,
            "cpi_yoy": 3.1,
            "ppi_yoy": 0.9,
            "policy_rate": 5.5,
        }
    ]).to_parquet(data_path, index=False)

    # Clone the repository definition while pointing its offline source at this fixture.
    view = macro_features.copy(deep=True)
    view.source = FileSource(
        name="test_macro_source",
        path=str(data_path),
        timestamp_field="event_timestamp",
        created_timestamp_column="ingested_at",
    )
    shutil.copy(Path("backend/feature_store.yaml"), tmp_path / "feature_store.yaml")
    store = FeatureStore(repo_path=str(tmp_path))
    store.apply([economy, view])

    entities = pd.DataFrame({
        "economy_code": ["USA", "USA"],
        "event_timestamp": pd.to_datetime([
            "2024-02-15T13:29:59Z", "2024-02-15T13:30:03Z"
        ], utc=True),
    })
    result = store.get_historical_features(
        entity_df=entities,
        features=["macro_features:cpi_yoy", "macro_features:policy_rate"],
    ).to_df()

    assert pd.isna(result.loc[0, "macro_features__cpi_yoy"])
    assert result.loc[1, "macro_features__cpi_yoy"] == pytest.approx(3.1)
    assert result.loc[1, "macro_features__policy_rate"] == pytest.approx(5.5)
