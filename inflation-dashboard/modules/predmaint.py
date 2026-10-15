"""Interactive, read-only visualization for PredMaint AI telemetry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

SENSOR_COLUMNS = ("vibration", "thermal", "acoustic")
THRESHOLDS = {
    "vibration": (7.1, 12.0, "mm/s RMS"),
    "thermal": (85.0, 105.0, "C"),
    "acoustic": (105.0, 125.0, "dB RMS"),
}
COLORS = {"vibration": "#38bdf8", "thermal": "#fb7185", "acoustic": "#a78bfa"}


def demo_telemetry() -> pd.DataFrame:
    """Return a deterministic local demo stream when no edge CSV is supplied."""
    generator = np.random.default_rng(42)
    time = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=360, freq="min")
    drift = np.linspace(0, 1, len(time))
    return pd.DataFrame({
        "timestamp": time,
        "vibration": 3.4 + 1.1 * drift + generator.normal(0, 0.35, len(time)),
        "thermal": 62 + 11 * drift + generator.normal(0, 1.6, len(time)),
        "acoustic": 80 + 9 * drift + generator.normal(0, 2.2, len(time)),
    })


def validate_telemetry(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", *SENSOR_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV must include: {', '.join(sorted(required))}")
    data = frame.loc[:, ["timestamp", *SENSOR_COLUMNS]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().sort_values("timestamp")
    if data.empty:
        raise ValueError("CSV has no valid timestamped sensor observations")
    return data


def feature_frame(data: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Calculate rolling condition indicators for interactive inspection."""
    features = pd.DataFrame({"timestamp": data["timestamp"]})
    for sensor in SENSOR_COLUMNS:
        rms = data[sensor].pow(2).rolling(window, min_periods=3).mean().pow(0.5)
        features[f"{sensor}_rms"] = rms
        features[f"{sensor}_kurtosis"] = data[sensor].rolling(window, min_periods=4).kurt() + 3
        features[f"{sensor}_crest_factor"] = data[sensor].abs().rolling(window, min_periods=3).max() / rms
    return features


def telemetry_figure(data: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True,
                        subplot_titles=[f"{sensor.title()} ({THRESHOLDS[sensor][2]})" for sensor in selected])
    for row, sensor in enumerate(selected, start=1):
        warning, trip, _ = THRESHOLDS[sensor]
        fig.add_trace(go.Scatter(
            x=data["timestamp"], y=data[sensor], name=sensor.title(), mode="lines",
            line={"color": COLORS[sensor], "width": 2}, hovertemplate="%{x|%H:%M UTC}<br>%{y:.2f}<extra></extra>",
        ), row=row, col=1)
        fig.add_hline(y=warning, line_dash="dot", line_color="#fbbf24", line_width=1,
                      annotation_text="Throttle", annotation_position="top left", row=row, col=1)
        fig.add_hline(y=trip, line_dash="dash", line_color="#ef4444", line_width=1,
                      annotation_text="Emergency stop", annotation_position="top left", row=row, col=1)
    fig.update_layout(title="Live sensor telemetry and hard safety thresholds", height=max(330, 245 * len(selected)),
                      template="plotly_dark", hovermode="x unified", showlegend=False,
                      margin={"l": 45, "r": 20, "t": 60, "b": 35})
    fig.update_xaxes(rangeslider_visible=len(selected) == 1, title_text="Timestamp (UTC)", row=len(selected), col=1)
    return fig


def condition_figure(features: pd.DataFrame, sensor: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=features[f"{sensor}_rms"], y=features[f"{sensor}_kurtosis"], mode="markers",
        marker={"size": 7, "color": features[f"{sensor}_crest_factor"], "colorscale": "Turbo", "showscale": True,
                "colorbar": {"title": "Crest factor"}},
        customdata=features[["timestamp", f"{sensor}_crest_factor"]],
        hovertemplate="%{customdata[0]|%Y-%m-%d %H:%M UTC}<br>RMS: %{x:.2f}<br>Pearson kurtosis: %{y:.2f}<br>Crest factor: %{customdata[1]:.2f}<extra></extra>",
    ))
    fig.update_layout(title=f"{sensor.title()} condition map", template="plotly_dark", height=410,
                      xaxis_title="Rolling RMS", yaxis_title="Rolling Pearson kurtosis",
                      margin={"l": 45, "r": 20, "t": 60, "b": 45})
    return fig


def render_predmaint_monitor() -> None:
    """Render read-only telemetry exploration. PLC actions remain backend-only."""
    st.caption("Read-only edge telemetry. Plot controls support hover, zoom, pan, and image export; PLC writes stay in the deterministic backend safety gate.")
    upload = st.file_uploader("Upload edge telemetry CSV", type="csv", key="iiot_csv",
                              help="Required columns: timestamp, vibration, thermal, acoustic.")
    try:
        data = validate_telemetry(pd.read_csv(upload)) if upload else demo_telemetry()
    except (ValueError, TypeError, pd.errors.ParserError) as error:
        st.error(str(error))
        return
    selected = st.multiselect("Sensor channels", SENSOR_COLUMNS, default=list(SENSOR_COLUMNS), key="iiot_channels")
    if not selected:
        st.info("Select at least one sensor channel.")
        return
    latest = data.iloc[-1]
    metrics = st.columns(3)
    for column, sensor in zip(metrics, SENSOR_COLUMNS):
        warning, trip, unit = THRESHOLDS[sensor]
        value = float(latest[sensor])
        state = "STOP" if value >= trip else "THROTTLE" if value >= warning else "NORMAL"
        column.metric(sensor.title(), f"{value:.2f} {unit}", state)
    config = {"displaylogo": False, "scrollZoom": True, "responsive": True}
    st.plotly_chart(telemetry_figure(data, selected), use_container_width=True, config=config)
    features = feature_frame(data)
    sensor = st.selectbox("Condition-map channel", selected, key="iiot_condition_sensor")
    st.plotly_chart(condition_figure(features, sensor), use_container_width=True, config=config)
