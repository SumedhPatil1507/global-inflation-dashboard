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
except (ImportError, OSError):
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


def autoencoder_anomalies(df: pd.DataFrame, epochs: int = 30, percentile: float = 95, contamination: float = None):
    if contamination is not None:
        percentile = max(1.0, min(99.9, (1.0 - float(contamination)) * 100.0))
    else:
        percentile = max(1.0, min(99.9, float(percentile)))

    if not TORCH_OK:
        df2, _ = zscore_anomalies(df, threshold=2.0)
        df2["recon_error"] = np.abs(df2["z_score"])
        df2["reconstruction_error"] = df2["recon_error"]
        thresh = float(df2["recon_error"].quantile(percentile / 100.0))
        df2["is_anomaly"] = df2["recon_error"] > thresh
        df2["_color"]     = ["red" if a else "steelblue" for a in df2["is_anomaly"]]
        return df2, thresh

    feat_cols = [c for c in ["interest_rate", "oil_price", "gdp_growth",
                              "unemployment_rate", "food_price_index", "inflation_rate"] if c in df.columns]
    if not feat_cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c not in ["year", "is_anomaly"]]
    
    if not feat_cols:
        # Fallback if no numeric feature columns found
        df2, _ = zscore_anomalies(df, threshold=2.0)
        df2["recon_error"] = np.abs(df2["z_score"])
        df2["reconstruction_error"] = df2["recon_error"]
        thresh = float(df2["recon_error"].quantile(percentile / 100.0))
        df2["is_anomaly"] = df2["recon_error"] > thresh
        df2["_color"]     = ["red" if a else "steelblue" for a in df2["is_anomaly"]]
        return df2, thresh

    sub    = df[feat_cols].fillna(0).values.astype(float)
    min_val = sub.min(0)
    max_val = sub.max(0)
    denom = max_val - min_val
    denom[denom == 0] = 1.0
    X_norm = (sub - min_val) / denom
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
        recon = model(X_t).detach().cpu().numpy()

    error  = np.mean((X_norm - recon) ** 2, axis=1)
    thresh = float(np.percentile(error, percentile))
    mask   = error > thresh
    colors = ["red" if a else "steelblue" for a in mask]
    return df.copy().assign(recon_error=error, reconstruction_error=error, is_anomaly=mask, _color=colors), thresh


def autoencoder_plot(df_ae: pd.DataFrame, threshold=None, anomalies: pd.DataFrame = None) -> go.Figure:
    fig = go.Figure()
    y_col = "recon_error" if "recon_error" in df_ae.columns else "reconstruction_error"
    x_col = "year" if "year" in df_ae.columns else df_ae.index
    color_col = df_ae["_color"].tolist() if "_color" in df_ae.columns else "steelblue"

    fig.add_trace(go.Scatter(x=df_ae[x_col] if isinstance(x_col, str) else x_col,
                             y=df_ae[y_col] if y_col in df_ae.columns else np.zeros(len(df_ae)),
                             mode="markers",
                             marker=dict(color=color_col, size=5, opacity=0.7),
                             name="Reconstruction Error"))

    thresh_val = None
    if isinstance(threshold, (int, float)):
        thresh_val = float(threshold)
    elif isinstance(threshold, pd.DataFrame) and not threshold.empty:
        if y_col in threshold.columns:
            thresh_val = float(threshold[y_col].min())
    elif isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        if y_col in anomalies.columns:
            thresh_val = float(anomalies[y_col].min())

    if thresh_val is not None:
        fig.add_hline(y=thresh_val, line_dash="dash", line_color="red",
                      annotation_text=f"Anomaly Threshold ({thresh_val:.4f})")
    fig.update_layout(title="Autoencoder Anomaly Detection (Reconstruction Error)",
                      xaxis_title="Year" if isinstance(x_col, str) else "Index",
                      yaxis_title="Reconstruction Error", height=420)
    return fig


# Backwards compatibility alias
autoencoder_loss_plot = autoencoder_plot

