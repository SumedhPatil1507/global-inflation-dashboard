"""
Anomaly detection: Z-score + Autoencoder.
Autoencoder falls back to Z-score proxy when torch is unavailable.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


def zscore_anomalies(df: pd.DataFrame, threshold: float = 3.0):
    df = df.copy()
    df["z_score"] = stats.zscore(df["inflation_rate"].fillna(df["inflation_rate"].mean()))
    return df, df[np.abs(df["z_score"]) > threshold]


def zscore_plot(df: pd.DataFrame, anomalies: pd.DataFrame, threshold: float = 3.0) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["inflation_rate"],
                             mode="markers", marker=dict(color="lightgray", size=4),
                             name="Normal"))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["year"], y=anomalies["inflation_rate"],
                                 mode="markers", marker=dict(color="red", size=8, symbol="x"),
                                 name=f"Anomaly (|Z|>{threshold:.1f})"))
    fig.update_layout(title="Z-Score Anomaly Detection — Inflation Rate",
                      xaxis_title="Year", yaxis_title="Inflation Rate (%)", height=420)
    return fig


def _build_autoencoder(in_dim: int):
    """Only called when TORCH_OK is True."""
    class _AE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, in_dim))
        def forward(self, x):
            return self.decoder(self.encoder(x))
    return _AE()


def autoencoder_anomalies(df: pd.DataFrame, epochs: int = 30, percentile: float = 95):
    if not TORCH_OK:
        df2, _ = zscore_anomalies(df, threshold=2.0)
        df2["recon_error"] = np.abs(df2["z_score"])
        thresh = float(df2["recon_error"].quantile(percentile / 100))
        df2["is_anomaly"] = df2["recon_error"] > thresh
        df2["_color"]     = ["red" if a else "steelblue" for a in df2["is_anomaly"]]
        return df2, thresh

    feat_cols = [c for c in ["interest_rate", "oil_price", "gdp_growth",
                              "unemployment_rate", "food_price_index"] if c in df.columns]
    sub    = df[feat_cols].fillna(0).values.astype(float)
    X_norm = (sub - sub.min(0)) / (sub.max(0) - sub.min(0) + 1e-8)
    X_t    = torch.tensor(X_norm, dtype=torch.float32)

    model = _build_autoencoder(len(feat_cols))
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        opt.zero_grad()
        recon = model(X_t)
        loss  = torch.nn.functional.mse_loss(recon, X_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        recon = model(X_t).numpy()

    error  = np.mean((X_norm - recon) ** 2, axis=1)
    thresh = float(np.percentile(error, percentile))
    mask   = error > thresh
    colors = ["red" if a else "steelblue" for a in mask]
    return df.copy().assign(recon_error=error, is_anomaly=mask, _color=colors), thresh


def autoencoder_plot(df_ae: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ae["year"], y=df_ae["recon_error"],
                             mode="markers",
                             marker=dict(color=df_ae["_color"].tolist(), size=5, opacity=0.7),
                             name="Reconstruction Error"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text="Anomaly Threshold")
    fig.update_layout(title="Autoencoder Anomaly Detection (Reconstruction Error)",
                      xaxis_title="Year", yaxis_title="Reconstruction Error", height=420)
    return fig
