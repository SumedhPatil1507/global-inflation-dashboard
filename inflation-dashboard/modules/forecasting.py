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
    # Build long-form directly from df_stress — avoids pivot/melt column naming issues
    fig = go.Figure()
    for asset in df_stress["Asset"].unique():
        sub = df_stress[df_stress["Asset"] == asset].sort_values("Year")
        fig.add_trace(go.Scatter(
            x=sub["Year"], y=sub["Real Return (%)"],
            mode="lines+markers", name=asset,
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text="Break-even")
    fig.update_layout(
        title=f"Portfolio Stress Test — {scenario_name}",
        xaxis_title="Year", yaxis_title="Real Return (%)",
        height=450, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def portfolio_bar(df_stress: pd.DataFrame) -> go.Figure:
    last_yr = df_stress["Year"].max()
    last    = df_stress[df_stress["Year"] == last_yr]
    fig = px.bar(last, x="Asset", y="Real Return (%)", color="Real Return (%)",
                 color_continuous_scale="RdYlGn",
                 title="Final Year Real Returns by Asset")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=380)
    return fig


def cumulative_wealth_plot(df_stress: pd.DataFrame, scenario_name: str) -> go.Figure:
    """Compound $1 invested in each asset over the horizon."""
    fig = go.Figure()
    for asset in df_stress["Asset"].unique():
        sub     = df_stress[df_stress["Asset"] == asset].sort_values("Year")
        returns = sub["Real Return (%)"].values / 100
        wealth  = np.cumprod(1 + returns)
        fig.add_trace(go.Scatter(
            x=sub["Year"].values, y=wealth,
            mode="lines+markers", name=asset,
            hovertemplate="%{y:.3f}x<extra>%{fullData.name}</extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="Initial $1")
    fig.update_layout(
        title=f"Cumulative Wealth ($1 Invested) — {scenario_name}",
        xaxis_title="Year", yaxis_title="Portfolio Value ($)",
        height=420, hovermode="x unified",
    )
    return fig


def sharpe_table(df_stress: pd.DataFrame, risk_free: float = 0.02) -> pd.DataFrame:
    """Compute annualised Sharpe ratio per asset."""
    rows = []
    for asset in df_stress["Asset"].unique():
        sub = df_stress[df_stress["Asset"] == asset].sort_values("Year")
        r   = sub["Real Return (%)"].values / 100
        mu  = np.mean(r)
        sd  = np.std(r) + 1e-9
        sharpe = (mu - risk_free) / sd
        max_dd = _max_drawdown(r)
        rows.append({
            "Asset":           asset,
            "Avg Real Return": f"{mu*100:.2f}%",
            "Volatility":      f"{sd*100:.2f}%",
            "Sharpe Ratio":    f"{sharpe:.2f}",
            "Max Drawdown":    f"{max_dd*100:.2f}%",
            "Signal":          "🟢 BUY" if sharpe > 0.5 else ("🔴 AVOID" if sharpe < 0 else "⚪ HOLD"),
        })
    return pd.DataFrame(rows).sort_values("Sharpe Ratio", ascending=False)


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(wealth)
    dd     = (wealth - peak) / (peak + 1e-9)
    return float(dd.min())


def monte_carlo_plot(base_return: float, volatility: float,
                     horizon: int = 10, simulations: int = 200,
                     asset_name: str = "Portfolio") -> go.Figure:
    """Monte Carlo simulation of asset return paths."""
    rng  = np.random.default_rng(42)
    fig  = go.Figure()
    final_vals = []
    for i in range(simulations):
        r      = rng.normal(base_return / 100, volatility / 100, horizon)
        wealth = np.cumprod(1 + r)
        final_vals.append(wealth[-1])
        color  = "rgba(56,189,248,0.08)"
        fig.add_trace(go.Scatter(
            x=list(range(1, horizon + 1)), y=wealth.tolist(),
            mode="lines", line=dict(color=color, width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Median path
    med_r  = rng.normal(base_return / 100, volatility / 100, (1000, horizon))
    median = np.median(np.cumprod(1 + med_r, axis=1), axis=0)
    fig.add_trace(go.Scatter(
        x=list(range(1, horizon + 1)), y=median.tolist(),
        mode="lines", name="Median Path",
        line=dict(color="#38bdf8", width=3),
    ))
    p5  = np.percentile(final_vals, 5)
    p95 = np.percentile(final_vals, 95)
    fig.update_layout(
        title=f"Monte Carlo ({simulations} paths) — {asset_name} | "
              f"P5: ${p5:.2f}  Median: ${np.median(final_vals):.2f}  P95: ${p95:.2f}",
        xaxis_title="Year", yaxis_title="Portfolio Value ($1 invested)",
        height=450,
    )
    return fig
