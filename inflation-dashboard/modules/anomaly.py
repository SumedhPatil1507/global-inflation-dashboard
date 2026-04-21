"""Anomaly detection: Z-score + Autoencoder."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats


def zscore_anomalies(df: pd.DataFrame, threshold: float = 3.0):
    df = df.copy()
    df["z_score"] = stats.zscore(df["inflation_rate"].fillna(df["inflation_rate"].mean()))
    anomalies = df[np.abs(df["z_score"]) > threshold]
    return df, anomalies


def zscore_plot(df: pd.DataFrame, anomalies: pd.DataFrame, threshold: float = 3.0) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["inflation_rate"],
                             mode="markers", marker=dict(color="lightgray", size=4),
                             name="Normal"))
    fig.add_trace(go.Scatter(x=anomalies["year"], y=anomalies["inflation_rate"],
                             mode="markers", marker=dict(color="red", size=8, symbol="x"),
                             name=f"Anomaly (|Z|>{threshold:.1f})"))
    fig.update_layout(title="Z-Score Anomaly Detection — Inflation Rate",
                      xaxis_title="Year", yaxis_title="Inflation Rate (%)", height=420)
    return fig


class _Autoencoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, in_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def autoencoder_anomalies(df: pd.DataFrame, epochs: int = 30, percentile: float = 95):
    feat_cols = [c for c in ["interest_rate", "oil_price", "gdp_growth",
                              "unemployment_rate", "food_price_index"] if c in df.columns]
    sub = df[feat_cols].fillna(0).values.astype(float)
    X_norm = (sub - sub.min(0)) / (sub.max(0) - sub.min(0) + 1e-8)
    X_t = torch.tensor(X_norm, dtype=torch.float32)

    model = _Autoencoder(len(feat_cols))
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss(reduction="none")

    for _ in range(epochs):
        opt.zero_grad()
        recon = model(X_t)
        loss = loss_fn(recon, X_t).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        recon = model(X_t).numpy()
    error = np.mean((X_norm - recon) ** 2, axis=1)
    threshold = np.percentile(error, percentile)
    anomaly_mask = error > threshold
    return df.copy().assign(recon_error=error, is_anomaly=anomaly_mask), threshold


def autoencoder_plot(df_ae: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ae["year"], y=df_ae["recon_error"],
                             mode="markers", marker=dict(
                                 color=np.where(df_ae["is_anomaly"], "red", "steelblue"),
                                 size=5, opacity=0.7),
                             name="Reconstruction Error"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text="Anomaly Threshold")
    fig.update_layout(title="Autoencoder Anomaly Detection (Reconstruction Error)",
                      xaxis_title="Year", yaxis_title="Reconstruction Error", height=420)
    return fig
