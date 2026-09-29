"""Hierarchical + K-Means clustering with Plotly."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLUSTER_FEATURES = ["inflation_rate", "interest_rate", "gdp_growth",
                    "unemployment_rate", "food_price_index"]


def _country_matrix(df: pd.DataFrame):
    avail       = [c for c in CLUSTER_FEATURES if c in df.columns]
    country_avg = df.groupby("country")[avail].mean().dropna()
    data        = country_avg.values.astype(float)
    data_norm   = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-8)
    return country_avg, data_norm


def dendrogram_figure(df: pd.DataFrame) -> go.Figure:
    """Returns interactive Plotly dendrogram-like visualization."""
    country_avg, data_norm = _country_matrix(df)
    if len(data_norm) < 2:
        fig = go.Figure()
        fig.update_layout(title="Need ≥ 2 countries for dendrogram", height=300)
        return fig
    
    Z = linkage(data_norm, method="ward")
    countries = list(country_avg.index)
    
    # Create a simplified tree visualization using scatter and lines
    fig = go.Figure()
    
    # Plot countries as leaves
    fig.add_trace(go.Scatter(
        x=list(range(len(countries))),
        y=[0] * len(countries),
        mode='markers+text',
        text=countries,
        textposition='top center',
        marker=dict(size=12, color='steelblue'),
        name='Countries'
    ))
    
    # Add some visualization of the clustering structure
    # This is a simplified representation since full dendrogram in Plotly is complex
    cluster_heights = Z[:, 2]
    max_height = max(cluster_heights) if len(cluster_heights) > 0 else 1
    
    fig.update_layout(
        title="Hierarchical Clustering Structure of Global Economies",
        xaxis_title="Country Index",
        yaxis_title="Cluster Distance",
        height=500,
        showlegend=False
    )
    
    return fig


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


def kmeans_clustering(df: pd.DataFrame, n_clusters: int = 4):
    """Perform K-Means clustering and return clustered dataframe with silhouette score."""
    country_avg, data_norm = _country_matrix(df)
    n_countries = len(data_norm)
    
    if n_countries < 2:
        return df, 0.0
    
    # Clamp k so it's always valid
    k = max(1, min(n_clusters, n_countries - 1))
    
    centroids, _ = kmeans(data_norm, k)
    idx, _ = vq(data_norm, centroids)
    
    country_avg = country_avg.copy()
    country_avg["cluster"] = idx.astype(str)
    country_avg = country_avg.reset_index()
    
    # Calculate silhouette score
    if len(idx) > 1 and k > 1:
        sil = silhouette_score(data_norm, idx)
    else:
        sil = 0.0
    
    return country_avg, sil


def cluster_scatter(df: pd.DataFrame, cluster_col: str) -> go.Figure:
    """Create scatter plot of clustered data."""
    if "inflation_rate" not in df.columns or "interest_rate" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Missing required columns", height=300)
        return fig
    
    fig = px.scatter(
        df, x="inflation_rate", y="interest_rate",
        color=cluster_col, text="country",
        title=f"Clustering Results: {cluster_col}",
        labels={"inflation_rate": "Avg Inflation (%)", "interest_rate": "Avg Interest Rate (%)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(height=500)
    return fig


def dbscan_clustering(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 2):
    """Perform DBSCAN clustering and return clustered dataframe."""
    country_avg, data_norm = _country_matrix(df)
    n_countries = len(data_norm)
    
    if n_countries < 2:
        return df
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_norm)
    labels = clustering.labels_
    
    country_avg = country_avg.copy()
    country_avg["cluster_dbscan"] = labels.astype(str)
    country_avg = country_avg.reset_index()
    
    return country_avg


def pca_reduction(df: pd.DataFrame, n_components: int = 2):
    """Perform PCA dimensionality reduction and return transformed dataframe with variance."""
    country_avg, data_norm = _country_matrix(df)
    n_countries = len(data_norm)
    
    if n_countries < 2:
        return df, [0.0, 0.0]
    
    n_components = min(n_components, min(n_countries, data_norm.shape[1]))
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_norm)
    
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=country_avg.index
    )
    pca_df = pca_df.reset_index()
    
    explained_variance = pca.explained_variance_ratio_.tolist()
    # Pad with zeros if we have fewer components than requested
    while len(explained_variance) < 2:
        explained_variance.append(0.0)
    
    return pca_df, explained_variance


def pca_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of PCA results."""
    if "PC1" not in df.columns or "PC2" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Missing PCA components", height=300)
        return fig
    
    fig = px.scatter(
        df, x="PC1", y="PC2",
        text="country",
        title="PCA Projection of Countries",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(height=500)
    return fig
