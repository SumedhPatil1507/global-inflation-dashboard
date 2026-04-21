
"""Advanced interactive visualizations: 3D, contour, hexbin, facet."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


def _has(df: pd.DataFrame, *cols) -> bool:
    """Return True only if every column exists and is non-empty."""
    return all(c in df.columns and df[c].notna().any() for c in cols)


def scatter_3d(df: pd.DataFrame) -> go.Figure:
    needed = ["oil_price", "food_price_index", "inflation_rate", "supply_chain_index"]
    if not _has(df, *needed):
        fig = go.Figure()
        fig.update_layout(title="3D plot unavailable — missing columns", height=300)
        return fig
    sample = df[needed].dropna().sample(min(3000, len(df)), random_state=42)
    fig = px.scatter_3d(
        sample, x="oil_price", y="food_price_index", z="inflation_rate",
        color="supply_chain_index", color_continuous_scale="Viridis",
        opacity=0.6, size_max=4,
        title="3D: Oil Price × Food Price Index × Inflation Rate",
        labels={
            "oil_price": "Oil Price", "food_price_index": "Food Price Index",
            "inflation_rate": "Inflation Rate (%)", "supply_chain_index": "Supply Chain Index",
        },
    )
    fig.update_layout(height=580)
    return fig


def contour_density(df: pd.DataFrame) -> go.Figure:
    if not _has(df, "oil_price", "interest_rate"):
        fig = go.Figure()
        fig.update_layout(title="Contour plot unavailable — missing columns", height=300)
        return fig
    sub = df[["oil_price", "interest_rate"]].dropna()
    fig = go.Figure(go.Histogram2dContour(
        x=sub["oil_price"], y=sub["interest_rate"],
        colorscale="Magma", reversescale=False,
        contours_coloring="fill", line_width=0,
    ))
    fig.update_layout(
        title="Contour Density: Oil Price vs Interest Rate",
        xaxis_title="Oil Price", yaxis_title="Interest Rate (%)", height=450,
    )
    return fig


def hexbin_plot(df: pd.DataFrame) -> str:
    """Returns base64 PNG — Plotly has no native hexbin."""
    fig, ax = plt.subplots(figsize=(9, 6))
    if _has(df, "money_supply_m2", "inflation_rate"):
        x = df["money_supply_m2"].fillna(0).values
        y = df["inflation_rate"].fillna(0).values
        hb = ax.hexbin(x, y, gridsize=40, cmap="plasma", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel("Money Supply M2")
        ax.set_ylabel("Inflation Rate (%)")
    else:
        ax.text(0.5, 0.5, "Hexbin unavailable — missing columns",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Hexbin Density: Money Supply M2 vs Inflation Rate")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def facet_inflation(df: pd.DataFrame, countries: list) -> go.Figure:
    if not countries or "inflation_rate" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Select at least one country", height=200)
        return fig
    sub = df[df["country"].isin(countries)].dropna(subset=["inflation_rate"]).sort_values("year")
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for selected countries", height=200)
        return fig
    fig = px.line(
        sub, x="year", y="inflation_rate",
        facet_col="country", facet_col_wrap=2,
        markers=True, color="country",
        title="Country-wise Inflation Trajectories (Facet Grid)",
        labels={"inflation_rate": "Inflation (%)", "year": "Year"},
    )
    fig.update_layout(height=500, showlegend=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
