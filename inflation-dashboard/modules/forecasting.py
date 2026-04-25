"""LSTM-based inflation forecasting per country."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


class _LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_sequences(series: np.ndarray, seq_len: int = 3):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def forecast_country(df: pd.DataFrame, country: str,
                     forecast_years: int = 5, epochs: int = 60, seq_len: int = 3):
    if not TORCH_OK:
        # Simple linear extrapolation fallback
        sub    = df[df["country"] == country].sort_values("year")
        series = sub["inflation_rate"].dropna().values.astype(np.float32)
        if len(series) < 2:
            return None, None, None, None
        trend      = np.polyfit(range(len(series)), series, 1)
        last_year  = int(sub["year"].max())
        future_yrs = list(range(last_year + 1, last_year + forecast_years + 1))
        future     = np.array([np.polyval(trend, len(series) + i) for i in range(forecast_years)])
        return series, sub["year"].values.tolist(), future, future_yrs
    sub    = df[df["country"] == country].sort_values("year")
    series = sub["inflation_rate"].dropna().values.astype(np.float32)

    if len(series) < seq_len + 2:
        return None, None, None, None

    mn, mx = series.min(), series.max()
    norm   = (series - mn) / (mx - mn + 1e-8)

    X, y = _make_sequences(norm, seq_len)
    # shapes: X → (N, seq_len, 1),  y → (N, 1)
    X_t = torch.from_numpy(X[:, :, None])   # (N, seq_len, 1)
    y_t = torch.from_numpy(y[:, None])       # (N, 1)

    model   = _LSTMModel()
    opt     = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()

    # Autoregressive forecast — window is always a plain Python list of floats
    window = list(norm[-seq_len:].astype(float))
    future_norm = []
    model.eval()
    with torch.no_grad():
        for _ in range(forecast_years):
            seq = np.array(window[-seq_len:], dtype=np.float32)
            inp = torch.from_numpy(seq[None, :, None])  # (1, seq_len, 1)
            pred = float(model(inp).item())
            future_norm.append(pred)
            window.append(pred)

    future       = np.array(future_norm) * (mx - mn) + mn
    last_year    = int(sub["year"].max())
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))
    return series, sub["year"].values.tolist(), future, future_years


def forecast_plot(series, hist_years, future, future_years, country: str) -> go.Figure:
    future     = np.asarray(future)
    std        = float(np.std(series))
    upper      = (future + std).tolist()
    lower      = (future - std).tolist()
    fy_forward = list(future_years)
    fy_reverse = list(reversed(future_years))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_years, y=series.tolist(),
        mode="lines+markers", name="Historical",
        line=dict(color="steelblue", width=2),
    ))
    # Confidence band — built from two plain Python lists
    fig.add_trace(go.Scatter(
        x=fy_forward + fy_reverse,
        y=upper + list(reversed(lower)),
        fill="toself", fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band (±1σ)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=fy_forward, y=future.tolist(),
        mode="lines+markers", name="LSTM Forecast",
        line=dict(color="orange", dash="dash", width=2),
    ))
    fig.update_layout(
        title=f"LSTM Inflation Forecast — {country}",
        xaxis_title="Year", yaxis_title="Inflation Rate (%)",
        height=450, hovermode="x unified",
    )
    return fig
