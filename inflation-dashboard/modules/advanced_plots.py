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


def scatter_3d(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter_3d(sample, x="oil_price", y="food_price_index", z="inflation_rate",
                        color="supply_chain_index", color_continuous_scale="Viridis",
                        opacity=0.6, size_max=4,
                        title="3D: Oil Price × Food Price Index × Inflation Rate",
                        labels={"oil_price": "Oil Price", "food_price_index": "Food Price Index",
                                "inflation_rate": "Inflation Rate (%)",
                                "supply_chain_index": "Supply Chain Index"})
    fig.update_layout(height=580)
    return fig


def contour_density(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram2dContour(
        x=df["oil_price"], y=df["interest_rate"],
        colorscale="Magma", reversescale=False,
        xaxis="x", yaxis="y", contours_coloring="fill",
        line_width=0,
    ))
    fig.update_layout(title="Contour Density: Oil Price vs Interest Rate",
                      xaxis_title="Oil Price", yaxis_title="Interest Rate (%)",
                      height=450)
    return fig


def hexbin_plot(df: pd.DataFrame) -> str:
    """Returns base64 PNG hexbin (Plotly doesn't support hexbin natively)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(df["money_supply_m2"].fillna(0),
                   df["inflation_rate"].fillna(0),
                   gridsize=40, cmap="plasma", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")
    ax.set_title("Hexbin Density: Money Supply M2 vs Inflation Rate")
    ax.set_xlabel("Money Supply M2")
    ax.set_ylabel("Inflation Rate (%)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def facet_inflation(df: pd.DataFrame, countries: list) -> go.Figure:
    sub = df[df["country"].isin(countries)].sort_values("year")
    fig = px.line(sub, x="year", y="inflation_rate", facet_col="country",
                  facet_col_wrap=2, markers=True,
                  title="Country-wise Inflation Trajectories (Facet Grid)",
                  labels={"inflation_rate": "Inflation (%)", "year": "Year"},
                  color="country")
    fig.update_layout(height=500, showlegend=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
