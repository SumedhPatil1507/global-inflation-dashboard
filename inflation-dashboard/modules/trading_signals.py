"""
Inflation-Adjusted Yield Optimizer & Regime Switching Allocator.
Generates actionable trading signals from macro data.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Regime detection ──────────────────────────────────────────────────────────
def detect_regime(inflation: float, gdp_growth: float, unemployment: float) -> dict:
    """
    Classify macro regime based on inflation, growth, unemployment.
    Returns regime label + recommended asset allocation.
    """
    if inflation > 6 and gdp_growth < 1:
        regime = "Stagflation"
        alloc  = {"Gold": 0.30, "Commodities": 0.25, "Real Estate": 0.20,
                  "Cash/T-Bills": 0.15, "Equities": 0.10, "Bonds (10Y)": 0.00}
        signal = "🔴 RISK-OFF — Reduce equities & bonds. Overweight real assets."
    elif inflation > 6 and gdp_growth >= 1:
        regime = "Overheating"
        alloc  = {"Commodities": 0.30, "Gold": 0.20, "Real Estate": 0.20,
                  "Equities": 0.20, "Cash/T-Bills": 0.10, "Bonds (10Y)": 0.00}
        signal = "🟡 CAUTION — Inflation running hot. Tilt toward commodities & real assets."
    elif inflation < 2 and gdp_growth < 0:
        regime = "Recession/Deflation"
        alloc  = {"Bonds (10Y)": 0.40, "Cash/T-Bills": 0.30, "Gold": 0.15,
                  "Equities": 0.15, "Commodities": 0.00, "Real Estate": 0.00}
        signal = "🔵 DEFENSIVE — Deflation risk. Overweight bonds & cash."
    elif 2 <= inflation <= 4 and gdp_growth >= 2:
        regime = "Goldilocks"
        alloc  = {"Equities": 0.50, "Real Estate": 0.20, "Bonds (10Y)": 0.15,
                  "Gold": 0.10, "Cash/T-Bills": 0.05, "Commodities": 0.00}
        signal = "🟢 RISK-ON — Ideal macro environment. Overweight equities."
    else:
        regime = "Transitional"
        alloc  = {"Equities": 0.35, "Bonds (10Y)": 0.25, "Real Estate": 0.15,
                  "Gold": 0.15, "Cash/T-Bills": 0.05, "Commodities": 0.05}
        signal = "⚪ NEUTRAL — Mixed signals. Balanced allocation."

    return {"regime": regime, "allocation": alloc, "signal": signal}


def regime_allocation_chart(alloc: dict, regime: str) -> go.Figure:
    fig = px.pie(
        names=list(alloc.keys()),
        values=list(alloc.values()),
        title=f"Recommended Allocation — {regime} Regime",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4,
    )
    fig.update_layout(height=400)
    return fig


# ── Inflation-adjusted yield optimizer ───────────────────────────────────────
def yield_optimizer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute real yield (nominal rate − inflation) per country.
    Rank countries by real yield for carry trade signals.
    """
    if "interest_rate" not in df.columns or "inflation_rate" not in df.columns:
        return pd.DataFrame()

    latest = df[df["year"] == df["year"].max()].copy()
    latest["real_yield"]    = latest["interest_rate"] - latest["inflation_rate"]
    latest["signal"]        = latest["real_yield"].apply(
        lambda x: "🟢 BUY" if x > 2 else ("🔴 SELL" if x < -1 else "⚪ HOLD")
    )
    latest["carry_rank"]    = latest["real_yield"].rank(ascending=False).astype(int)
    return latest[["country", "interest_rate", "inflation_rate",
                   "real_yield", "signal", "carry_rank"]].sort_values("carry_rank")


def yield_optimizer_chart(df_yield: pd.DataFrame) -> go.Figure:
    if df_yield.empty:
        return go.Figure()
    colors = df_yield["real_yield"].apply(
        lambda x: "#22c55e" if x > 2 else ("#ef4444" if x < -1 else "#94a3b8")
    ).tolist()
    fig = go.Figure(go.Bar(
        x=df_yield["country"], y=df_yield["real_yield"],
        marker_color=colors, text=df_yield["signal"],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    fig.update_layout(
        title="Inflation-Adjusted Real Yield by Country (Carry Trade Signals)",
        xaxis_title="Country", yaxis_title="Real Yield (%)",
        height=420,
    )
    return fig


# ── Country signal table ──────────────────────────────────────────────────────
def signal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Full signal table with regime + yield signal per country."""
    if df.empty:
        return pd.DataFrame()
    latest = df[df["year"] == df["year"].max()].copy()
    rows   = []
    for _, row in latest.iterrows():
        inf  = row.get("inflation_rate", 0) or 0
        gdp  = row.get("gdp_growth", 2) or 2
        unemp= row.get("unemployment_rate", 5) or 5
        r    = detect_regime(inf, gdp, unemp)
        rows.append({
            "Country":        row["country"],
            "Inflation (%)":  round(inf, 2),
            "GDP Growth (%)": round(gdp, 2),
            "Regime":         r["regime"],
            "Signal":         r["signal"],
        })
    return pd.DataFrame(rows).sort_values("Inflation (%)", ascending=False)
