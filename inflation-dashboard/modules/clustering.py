"""Hierarchical + K-Means clustering with Plotly."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.vq import kmeans, vq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

CLUSTER_FEATURES = ["inflation_rate", "interest_rate", "gdp_growth",
                    "unemployment_rate", "food_price_index"]


def _country_matrix(df: pd.DataFrame):
    avail       = [c for c in CLUSTER_FEATURES if c in df.columns]
    country_avg = df.groupby("country")[avail].mean().dropna()
    data        = country_avg.values.astype(float)
    data_norm   = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-8)
    return country_avg, data_norm


def dendrogram_figure(df: pd.DataFrame) -> str:
    """Returns base64 PNG — scipy dendrogram has no Plotly equivalent."""
    country_avg, data_norm = _country_matrix(df)
    if len(data_norm) < 2:
        # Not enough countries — return blank image
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Need ≥ 2 countries for dendrogram",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        Z = linkage(data_norm, method="ward")
        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(Z, labels=list(country_avg.index), leaf_rotation=90,
                   leaf_font_size=9, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title("Hierarchical Clustering Dendrogram of Global Economies")
        ax.set_xlabel("Country")
        ax.set_ylabel("Distance")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def elbow_plot(df: pd.DataFrame, max_k: int = 10) -> go.Figure:
    _, data_norm = _country_matrix(df)
    max_k        = min(max_k, len(data_norm) - 1)
    distortions  = []
    for k in range(1, max_k + 1):
        _, dist = kmeans(data_norm, k)
        distortions.append(float(dist))
    fig = px.line(x=list(range(1, len(distortions) + 1)), y=distortions,
                  markers=True,
                  labels={"x": "Number of Clusters (k)", "y": "Distortion"},
                  title="Elbow Method — Optimal Number of Clusters")
    fig.update_layout(height=380)
    return fig


def kmeans_scatter(df: pd.DataFrame, k: int = 4) -> go.Figure:
    country_avg, data_norm = _country_matrix(df)
    n_countries = len(data_norm)

    if n_countries < 2:
        fig = go.Figure()
        fig.update_layout(title="Need ≥ 2 countries for clustering", height=300)
        return fig

    # Clamp k so it's always valid
    k = max(1, min(k, n_countries - 1))

    centroids, _ = kmeans(data_norm, k)
    idx, _       = vq(data_norm, centroids)
    country_avg  = country_avg.copy()
    country_avg["cluster"] = idx.astype(str)
    country_avg  = country_avg.reset_index()

    fig = px.scatter(
        country_avg, x="inflation_rate", y="interest_rate",
        color="cluster", text="country",
        title=f"K-Means Clusters (k={k}): Inflation vs Interest Rate",
        labels={"inflation_rate": "Avg Inflation (%)", "interest_rate": "Avg Interest Rate (%)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(height=500)
    return fig
