"""
LSTM forecasting + Portfolio Stress Tester (Stagflation scenario).
Falls back to linear extrapolation when torch is unavailable.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


def _build_lstm():
    """Build LSTM model — only called when TORCH_OK is True."""
    class _LSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
            self.fc   = torch.nn.Linear(64, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    return _LSTM()


def _make_sequences(series: np.ndarray, seq_len: int = 3):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _linear_forecast(df, country, forecast_years):
    sub    = df[df["country"] == country].sort_values("year")
    series = sub["inflation_rate"].dropna().values.astype(np.float32)
    if len(series) < 2:
        return None, None, None, None
    trend     = np.polyfit(range(len(series)), series, 1)
    last_year = int(sub["year"].max())
    fut_yrs   = list(range(last_year + 1, last_year + forecast_years + 1))
    future    = np.array([np.polyval(trend, len(series) + i) for i in range(forecast_years)])
    return series, sub["year"].values.tolist(), future, fut_yrs


def forecast_country(df: pd.DataFrame, country: str,
                     forecast_years: int = 5, epochs: int = 60, seq_len: int = 3):
    if not TORCH_OK:
        return _linear_forecast(df, country, forecast_years)

    sub    = df[df["country"] == country].sort_values("year")
    series = sub["inflation_rate"].dropna().values.astype(np.float32)
    if len(series) < seq_len + 2:
        return _linear_forecast(df, country, forecast_years)

    mn, mx = series.min(), series.max()
    norm   = (series - mn) / (mx - mn + 1e-8)
    X, y   = _make_sequences(norm, seq_len)
    X_t    = torch.from_numpy(X[:, :, None])
    y_t    = torch.from_numpy(y[:, None])

    model   = _build_lstm()
    opt     = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()

    window = list(norm[-seq_len:].astype(float))
    future_norm = []
    model.eval()
    with torch.no_grad():
        for _ in range(forecast_years):
            seq = np.array(window[-seq_len:], dtype=np.float32)
            inp = torch.from_numpy(seq[None, :, None])
            future_norm.append(float(model(inp).item()))
            window.append(future_norm[-1])

    future    = np.array(future_norm) * (mx - mn) + mn
    last_year = int(sub["year"].max())
    fut_yrs   = list(range(last_year + 1, last_year + forecast_years + 1))
    return series, sub["year"].values.tolist(), future, fut_yrs


def forecast_plot(series, hist_years, future, future_years, country: str) -> go.Figure:
    future = np.asarray(future)
    std    = float(np.std(series))
    upper  = (future + std).tolist()
    lower  = (future - std).tolist()
    fwd    = list(future_years)
    rev    = list(reversed(future_years))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_years, y=series.tolist(),
                             mode="lines+markers", name="Historical",
                             line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=fwd + rev, y=upper + list(reversed(lower)),
                             fill="toself", fillcolor="rgba(255,165,0,0.15)",
                             line=dict(color="rgba(0,0,0,0)"),
                             name="Confidence Band (±1σ)"))
    fig.add_trace(go.Scatter(x=fwd, y=future.tolist(),
                             mode="lines+markers", name="Forecast",
                             line=dict(color="orange", dash="dash", width=2)))
    fig.update_layout(title=f"Inflation Forecast — {country}",
                      xaxis_title="Year", yaxis_title="Inflation Rate (%)",
                      height=450, hovermode="x unified")
    return fig


# ── Portfolio Stress Tester ───────────────────────────────────────────────────
ASSETS = {
    "Equities":       {"real_return_base": 0.07,  "inflation_sensitivity": -0.40},
    "Bonds (10Y)":    {"real_return_base": 0.03,  "inflation_sensitivity": -0.80},
    "Gold":           {"real_return_base": 0.02,  "inflation_sensitivity":  0.60},
    "Real Estate":    {"real_return_base": 0.05,  "inflation_sensitivity":  0.20},
    "Cash/T-Bills":   {"real_return_base": 0.01,  "inflation_sensitivity": -0.30},
    "Commodities":    {"real_return_base": 0.04,  "inflation_sensitivity":  0.70},
}

SCENARIOS = {
    "Stagflation (High Inflation + High Unemployment)": {"inflation_shock": 8.0,  "unemployment_shock": 5.0},
    "Hyperinflation":                                   {"inflation_shock": 20.0, "unemployment_shock": 2.0},
    "Deflation":                                        {"inflation_shock": -3.0, "unemployment_shock": 3.0},
    "Soft Landing":                                     {"inflation_shock": 2.0,  "unemployment_shock": 0.5},
    "Custom":                                           {"inflation_shock": 0.0,  "unemployment_shock": 0.0},
}


def run_stress_test(base_inflation: float, scenario_name: str,
                    custom_inf: float = 0.0, custom_unemp: float = 0.0,
                    horizon: int = 10, portfolio: dict = None) -> pd.DataFrame:
    """
    Simulate asset real returns under a macro stress scenario.
    portfolio: {asset_name: weight} — weights must sum to 1.
    """
    if portfolio is None:
        portfolio = {a: 1 / len(ASSETS) for a in ASSETS}

    params = SCENARIOS[scenario_name].copy()
    if scenario_name == "Custom":
        params = {"inflation_shock": custom_inf, "unemployment_shock": custom_unemp}

    shocked_inf = base_inflation + params["inflation_shock"]
    rows = []
    for year in range(1, horizon + 1):
        # Inflation decays toward base over horizon
        inf_t = shocked_inf - (shocked_inf - base_inflation) * (year / horizon)
        port_return = 0.0
        for asset, meta in ASSETS.items():
            w = portfolio.get(asset, 0)
            real_ret = meta["real_return_base"] + meta["inflation_sensitivity"] * (inf_t / 100)
            port_return += w * real_ret
            rows.append({"Year": year, "Asset": asset,
                         "Inflation (%)": round(inf_t, 2),
                         "Real Return (%)": round(real_ret * 100, 2),
                         "Weight": w})
    return pd.DataFrame(rows)


def stress_test_plot(df_stress: pd.DataFrame, scenario_name: str) -> go.Figure:
    pivot = df_stress.pivot_table(index="Year", columns="Asset",
                                  values="Real Return (%)", aggfunc="mean")
    fig = px.line(pivot.reset_index().melt(id_vars="Year"),
                  x="Year", y="value", color="variable",
                  title=f"Portfolio Stress Test — {scenario_name}",
                  labels={"value": "Real Return (%)", "variable": "Asset"})
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text="Break-even")
    fig.update_layout(height=450, hovermode="x unified")
    return fig


def portfolio_bar(df_stress: pd.DataFrame) -> go.Figure:
    last_yr = df_stress["Year"].max()
    last    = df_stress[df_stress["Year"] == last_yr]
    fig = px.bar(last, x="Asset", y="Real Return (%)", color="Real Return (%)",
                 color_continuous_scale="RdYlGn",
                 title=f"Final Year Real Returns by Asset")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=380)
    return fig
