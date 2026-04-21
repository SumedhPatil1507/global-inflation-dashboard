"""LSTM-based inflation forecasting per country."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim


class _LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_sequences(series: np.ndarray, seq_len: int = 3):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)


def forecast_country(df: pd.DataFrame, country: str,
                     forecast_years: int = 5, epochs: int = 60, seq_len: int = 3):
    sub = df[df["country"] == country].sort_values("year")
    series = sub["inflation_rate"].dropna().values.astype(float)

    if len(series) < seq_len + 2:
        return None, None, None

    mn, mx = series.min(), series.max()
    norm = (series - mn) / (mx - mn + 1e-8)

    X, y = _make_sequences(norm, seq_len)
    X_t = torch.tensor(X[:, :, None], dtype=torch.float32)
    y_t = torch.tensor(y[:, None], dtype=torch.float32)

    model = _LSTMModel()
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()

    # Forecast future years
    window = list(norm[-seq_len:])
    future_norm = []
    model.eval()
    with torch.no_grad():
        for _ in range(forecast_years):
            inp = torch.tensor([[window[-seq_len:]]], dtype=torch.float32).squeeze(0)
            inp = inp.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            pred = model(inp).item()
            future_norm.append(pred)
            window.append(pred)

    future = np.array(future_norm) * (mx - mn) + mn
    last_year = int(sub["year"].max())
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))
    return series, sub["year"].values, future, future_years


def forecast_plot(series, hist_years, future, future_years, country: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(hist_years), y=list(series),
                             mode="lines+markers", name="Historical",
                             line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=future_years, y=list(future),
                             mode="lines+markers", name="LSTM Forecast",
                             line=dict(color="orange", dash="dash")))
    fig.update_layout(title=f"LSTM Inflation Forecast — {country}",
                      xaxis_title="Year", yaxis_title="Inflation Rate (%)",
                      height=420)
    return fig
