"""Interactive EDA plots using Plotly."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def histogram_grid(df: pd.DataFrame) -> go.Figure:
    cols = ["inflation_rate", "interest_rate", "oil_price",
            "gdp_growth", "unemployment_rate", "food_price_index"]
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[c.replace("_", " ").title() for c in cols])
    for idx, col in enumerate(cols):
        r, c = divmod(idx, 3)
        fig.add_trace(go.Histogram(x=df[col], nbinsx=40, name=col,
                                   marker_color="steelblue", opacity=0.75),
                      row=r + 1, col=c + 1)
    fig.update_layout(title="Distribution of Key Economic Variables",
                      showlegend=False, height=500)
    return fig


def line_trends(df: pd.DataFrame, countries: list, metric: str) -> go.Figure:
    fig = go.Figure()
    for country in countries:
        sub = df[df["country"] == country].sort_values("year")
        fig.add_trace(go.Scatter(x=sub["year"], y=sub[metric],
                                 mode="lines+markers", name=country))
    fig.update_layout(title=f"{metric.replace('_', ' ').title()} Trend by Country",
                      xaxis_title="Year", yaxis_title=metric, height=420)
    return fig


def avg_inflation_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    avg = (df.groupby("country")["inflation_rate"]
             .mean().sort_values(ascending=False).head(top_n).reset_index())
    fig = px.bar(avg, x="inflation_rate", y="country", orientation="h",
                 color="inflation_rate", color_continuous_scale="Viridis",
                 title=f"Top {top_n} Countries by Avg Inflation Rate",
                 labels={"inflation_rate": "Avg Inflation (%)", "country": ""})
    fig.update_layout(height=420)
    return fig


def region_pie(df: pd.DataFrame) -> go.Figure:
    region_map = {
        "USA": "Developed", "EU": "Developed", "JPN": "Developed", "GBR": "Developed",
        "CAN": "Developed", "AUS": "Developed", "KOR": "Developed",
        "IND": "Emerging", "CHN": "Emerging", "BRA": "Emerging", "RUS": "Emerging",
        "MEX": "Emerging", "IDN": "Emerging", "THA": "Emerging", "VNM": "Emerging",
        "PHL": "Emerging", "ZAF": "Emerging", "SAU": "Emerging",
        "ARG": "Frontier", "TUR": "Frontier",
    }
    df = df.copy()
    df["region"] = df["country"].map(region_map).fillna("Other")
    counts = df["region"].value_counts().reset_index()
    counts.columns = ["region", "count"]
    fig = px.pie(counts, names="region", values="count",
                 title="Data Distribution by Economic Region",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    return fig


def boxplot_inflation(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="country", y="inflation_rate",
                 color="country", title="Inflation Rate Distribution by Country",
                 labels={"inflation_rate": "Inflation Rate (%)"})
    fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=480)
    return fig


def violin_by_year(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(df, x="year", y="inflation_rate", color="year",
                    box=True, points=False,
                    title="Inflation Distribution by Year (Violin)",
                    labels={"inflation_rate": "Inflation Rate (%)"})
    fig.update_layout(showlegend=False, height=420)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = ["inflation_rate", "interest_rate", "oil_price", "gdp_growth",
                "unemployment_rate", "food_price_index", "supply_chain_index"]
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap", aspect="auto")
    fig.update_layout(height=500)
    return fig


def scatter_matrix(df: pd.DataFrame) -> go.Figure:
    cols = ["inflation_rate", "interest_rate", "oil_price", "gdp_growth", "unemployment_rate"]
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter_matrix(sample, dimensions=cols, color="year",
                            title="Scatter Matrix — Multivariate Relationships",
                            opacity=0.5)
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=600)
    return fig
