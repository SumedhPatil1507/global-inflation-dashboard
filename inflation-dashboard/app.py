"""
🌍 Global Inflation Insights Dashboard
Streamlit App — Live World Bank Data + ML/DL Analytics
"""
import base64
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Inflation Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h2 { color: #38bdf8; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #94a3b8; font-size: 0.85rem; margin: 0; }
    .section-header {
        font-size: 1.4rem; font-weight: 700;
        color: #38bdf8; margin-top: 1.5rem; margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Lazy module imports (avoid top-level torch import delays) ─────────────────
from modules.data_loader import get_data
from modules import eda, anomaly, clustering, forecasting, advanced_plots
from modules.models import (train_and_evaluate, actual_vs_predicted_plot,
                             loss_curve_plot, permutation_importance)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Inflation Dashboard")
    st.markdown("---")

    data_source = st.radio("Data Source", ["Live (World Bank API)", "Synthetic Demo"],
                           index=0)
    source_key = "live" if "Live" in data_source else "synthetic"

    st.markdown("### 📅 Year Range")
    year_range = st.slider("Select years", 2015, 2024, (2020, 2024))

    st.markdown("### 🌐 Countries")
    all_countries_placeholder = st.empty()  # filled after data loads

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    epochs = st.slider("Training Epochs", 10, 100, 40, step=10)
    forecast_years = st.slider("Forecast Horizon (years)", 1, 10, 5)
    n_clusters = st.slider("K-Means Clusters", 2, 8, 4)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + PyTorch + World Bank API")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching data…"):
    df_raw = get_data(source=source_key,
                      start_year=year_range[0], end_year=year_range[1])

all_countries = sorted(df_raw["country"].unique().tolist())
default_sel = [c for c in ["USA", "IND", "CHN", "EU", "GBR", "BRA"] if c in all_countries]

with st.sidebar:
    selected_countries = all_countries_placeholder.multiselect(
        "Filter countries", all_countries, default=default_sel or all_countries[:6]
    )

df = df_raw[df_raw["country"].isin(selected_countries)] if selected_countries else df_raw

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Global Inflation Insights Dashboard")
st.markdown(
    "Post-COVID economic analysis across major economies — "
    "live data, ML models, anomaly detection & forecasting."
)
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
avg_inf  = df["inflation_rate"].mean()
max_inf  = df["inflation_rate"].max()
min_inf  = df["inflation_rate"].min()
n_ctries = df["country"].nunique()

c1, c2, c3, c4 = st.columns(4)
for col, label, val, fmt in [
    (c1, "Avg Inflation", avg_inf, "{:.2f}%"),
    (c2, "Peak Inflation", max_inf, "{:.2f}%"),
    (c3, "Lowest Inflation", min_inf, "{:.2f}%"),
    (c4, "Countries", n_ctries, "{:.0f}"),
]:
    col.markdown(
        f'<div class="metric-card"><h2>{fmt.format(val)}</h2><p>{label}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 EDA",
    "🤖 ML Models",
    "🔍 Anomaly Detection",
    "🔬 Clustering",
    "📈 Forecasting",
    "🎨 Advanced Plots",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-header">Distribution of Key Variables</p>',
                unsafe_allow_html=True)
    st.plotly_chart(eda.histogram_grid(df), use_container_width=True)

    st.markdown('<p class="section-header">Time-Series Trends</p>',
                unsafe_allow_html=True)
    metric = st.selectbox("Metric", ["inflation_rate", "interest_rate",
                                     "gdp_growth", "unemployment_rate"], key="ts_metric")
    st.plotly_chart(eda.line_trends(df, selected_countries or all_countries[:6], metric),
                    use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">Top Countries by Avg Inflation</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(eda.avg_inflation_bar(df), use_container_width=True)
    with col_b:
        st.markdown('<p class="section-header">Region Composition</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(eda.region_pie(df), use_container_width=True)

    st.markdown('<p class="section-header">Inflation Boxplot by Country</p>',
                unsafe_allow_html=True)
    st.plotly_chart(eda.boxplot_inflation(df), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<p class="section-header">Violin Plot by Year</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(eda.violin_by_year(df), use_container_width=True)
    with col_d:
        st.markdown('<p class="section-header">Correlation Heatmap</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)

    st.markdown('<p class="section-header">Scatter Matrix</p>',
                unsafe_allow_html=True)
    st.plotly_chart(eda.scatter_matrix(df), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML Models
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">Neural Network — Inflation Predictor</p>',
                unsafe_allow_html=True)
    st.info("Trains a 3-layer feedforward network on the filtered dataset.")

    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner(f"Training for {epochs} epochs…"):
            preds, actual, losses, mse, r2, model, X_t, feat_names = \
                train_and_evaluate(df, epochs=epochs)

        y_t = torch.tensor(actual.reshape(-1, 1), dtype=torch.float32)

        m1, m2 = st.columns(2)
        m1.metric("MSE", f"{mse:.4f}")
        m2.metric("R² Score", f"{r2:.4f}")

        col_e, col_f = st.columns(2)
        with col_e:
            st.plotly_chart(actual_vs_predicted_plot(actual, preds),
                            use_container_width=True)
        with col_f:
            st.plotly_chart(loss_curve_plot(losses), use_container_width=True)

        st.markdown('<p class="section-header">Feature Importance</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(permutation_importance(model, X_t, y_t, feat_names),
                        use_container_width=True)
    else:
        st.markdown("👆 Click **Train Model** to run the neural network.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-header">Z-Score Anomaly Detection</p>',
                unsafe_allow_html=True)
    z_thresh = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.5, key="z_thresh")
    df_z, anom_z = anomaly.zscore_anomalies(df, threshold=z_thresh)
    st.plotly_chart(anomaly.zscore_plot(df_z, anom_z, threshold=z_thresh), use_container_width=True)
    st.caption(f"Detected **{len(anom_z)}** anomalous observations out of {len(df_z)}.")

    st.markdown('<p class="section-header">Autoencoder Anomaly Detection</p>',
                unsafe_allow_html=True)
    ae_pct = st.slider("Anomaly Percentile Threshold", 80, 99, 95, key="ae_pct")

    if st.button("🔍 Run Autoencoder", key="ae_btn"):
        with st.spinner("Training autoencoder…"):
            df_ae, ae_thresh = anomaly.autoencoder_anomalies(df, percentile=ae_pct)
        st.plotly_chart(anomaly.autoencoder_plot(df_ae, ae_thresh),
                        use_container_width=True)
        n_ae = df_ae["is_anomaly"].sum()
        st.caption(f"Detected **{n_ae}** anomalies (top {100 - ae_pct}% reconstruction error).")
    else:
        st.markdown("👆 Click **Run Autoencoder** to detect anomalies.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Clustering
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">Hierarchical Clustering Dendrogram</p>',
                unsafe_allow_html=True)
    dend_b64 = clustering.dendrogram_figure(df)
    st.image(base64.b64decode(dend_b64), use_container_width=True)

    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown('<p class="section-header">Elbow Method</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(clustering.elbow_plot(df), use_container_width=True)
    with col_h:
        st.markdown(f'<p class="section-header">K-Means (k={n_clusters})</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(clustering.kmeans_scatter(df, k=n_clusters),
                        use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Forecasting
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-header">LSTM Inflation Forecasting</p>',
                unsafe_allow_html=True)
    forecast_country = st.selectbox("Select Country", all_countries,
                                    index=all_countries.index("USA") if "USA" in all_countries else 0,
                                    key="fc_country")

    if st.button("📈 Run Forecast", key="fc_btn"):
        with st.spinner(f"Training LSTM for {forecast_country}…"):
            series, hist_years, future, future_years = forecasting.forecast_country(
                df_raw, forecast_country, forecast_years=forecast_years
            )
        if series is not None:
            st.plotly_chart(
                forecasting.forecast_plot(series, hist_years, future, future_years,
                                          forecast_country),
                use_container_width=True,
            )
            fc_df = pd.DataFrame({"Year": future_years,
                                  "Forecasted Inflation (%)": np.round(future, 3)})
            st.dataframe(fc_df, use_container_width=True)
        else:
            st.warning("Not enough data for this country. Try another.")
    else:
        st.markdown("👆 Select a country and click **Run Forecast**.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Advanced Plots
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-header">3D Scatter: Oil × Food × Inflation</p>',
                unsafe_allow_html=True)
    st.plotly_chart(advanced_plots.scatter_3d(df), use_container_width=True)

    col_i, col_j = st.columns(2)
    with col_i:
        st.markdown('<p class="section-header">Contour Density: Oil vs Interest Rate</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(advanced_plots.contour_density(df), use_container_width=True)
    with col_j:
        st.markdown('<p class="section-header">Hexbin: Money Supply vs Inflation</p>',
                    unsafe_allow_html=True)
        hb_b64 = advanced_plots.hexbin_plot(df)
        st.image(base64.b64decode(hb_b64), use_container_width=True)

    st.markdown('<p class="section-header">Facet Grid: Country Inflation Trajectories</p>',
                unsafe_allow_html=True)
    facet_sel = st.multiselect("Countries for Facet", all_countries,
                               default=(selected_countries or all_countries)[:4],
                               key="facet_sel")
    if facet_sel:
        st.plotly_chart(advanced_plots.facet_inflation(df, facet_sel),
                        use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#475569;font-size:0.8rem;'>"
    "Global Inflation Insights Dashboard · Data: World Bank API · "
    "Built with Streamlit, PyTorch, Plotly"
    "</center>",
    unsafe_allow_html=True,
)
