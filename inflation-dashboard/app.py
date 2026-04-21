
"""
🌍 Global Inflation Insights Dashboard
Premium Streamlit App — Live World Bank Data + ML/DL Analytics
Features: Auth, Usage Logs, Cost Tracking, Feedback, Webhooks, Confidence Scores,
          Citations, Editable Data, Anomaly Detection, Clustering, LSTM Forecasting
"""
import base64
import streamlit as st
import pandas as pd
import numpy as np
import torch

# ── Page config ───────────────────────────────────────────────────────────────
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
    background: linear-gradient(135deg,#1e293b,#0f172a);
    border:1px solid #334155; border-radius:12px;
    padding:1.2rem 1.5rem; text-align:center;
}
.metric-card h2 { color:#38bdf8; font-size:2rem; margin:0; }
.metric-card p  { color:#94a3b8; font-size:0.85rem; margin:0; }
.section-header { font-size:1.3rem; font-weight:700; color:#38bdf8;
                  margin-top:1.2rem; margin-bottom:0.4rem; }
.citation-box { background:#1e293b; border-left:4px solid #38bdf8;
                padding:0.6rem 1rem; border-radius:6px;
                font-size:0.8rem; color:#94a3b8; margin-top:0.5rem; }
.confidence-badge { display:inline-block; padding:2px 10px; border-radius:20px;
                    font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Module imports ────────────────────────────────────────────────────────────
from modules.security     import login_wall, logout_button, current_user
from modules.usage_logger import log_action, submit_feedback, render_admin_panel
from modules.webhooks     import fire, render_webhook_settings
from modules.data_loader  import get_data
from modules              import eda, anomaly, clustering, forecasting, advanced_plots
from modules.models       import (train_and_evaluate, actual_vs_predicted_plot,
                                   loss_curve_plot, permutation_importance)

# ── Auth wall ─────────────────────────────────────────────────────────────────
login_wall()
user = current_user()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🌍 Inflation Dashboard")
    st.markdown(f"👤 Logged in as **{user}**")
    logout_button()
    st.markdown("---")

    data_source = st.radio("Data Source", ["Live (World Bank API)", "Synthetic Demo"], index=0)
    source_key  = "live" if "Live" in data_source else "synthetic"

    st.markdown("### 📅 Year Range")
    year_range = st.slider("Select years", 2015, 2024, (2020, 2024))

    st.markdown("### 🌐 Countries")
    country_placeholder = st.empty()

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    epochs         = st.slider("Training Epochs",       10, 100, 40, step=10)
    forecast_years = st.slider("Forecast Horizon (yrs)", 1,  10,  5)
    n_clusters     = st.slider("K-Means Clusters",       2,   8,  4)

    st.markdown("---")
    render_webhook_settings()
    st.markdown("---")
    st.caption("Built with Streamlit · PyTorch · World Bank API")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching data…"):
    df_raw = get_data(source=source_key, start_year=year_range[0], end_year=year_range[1])

all_countries  = sorted(df_raw["country"].unique().tolist())
default_sel    = [c for c in ["USA","IND","CHN","EU","GBR","BRA"] if c in all_countries]

with st.sidebar:
    selected_countries = country_placeholder.multiselect(
        "Filter countries", all_countries, default=default_sel or all_countries[:6]
    )

df = df_raw[df_raw["country"].isin(selected_countries)] if selected_countries else df_raw
log_action(user, "data_load", f"source={source_key} rows={len(df)}", rows=len(df))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Global Inflation Insights Dashboard")
st.markdown(
    "Post-COVID economic analysis across major economies — "
    "live data, ML models, anomaly detection & forecasting."
)

# Citation
st.markdown(
    '<div class="citation-box">📚 <b>Data Source:</b> '
    '<a href="https://data.worldbank.org" target="_blank">World Bank Open Data</a> — '
    'CPI Inflation (FP.CPI.TOTL.ZG), GDP Growth (NY.GDP.MKTP.KD.ZG), '
    'Unemployment (SL.UEM.TOTL.ZS), Lending Rate (FR.INR.LEND). '
    'Retrieved via <a href="https://pypi.org/project/wbgapi/" target="_blank">wbgapi</a>. '
    'Supplementary columns (oil price, food index, M2, supply chain) are synthetic proxies.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
avg_inf  = df["inflation_rate"].mean()
max_inf  = df["inflation_rate"].max()
min_inf  = df["inflation_rate"].min()
n_ctries = df["country"].nunique()

for col, label, val, fmt in zip(
    st.columns(4),
    ["Avg Inflation", "Peak Inflation", "Lowest Inflation", "Countries"],
    [avg_inf, max_inf, min_inf, n_ctries],
    ["{:.2f}%", "{:.2f}%", "{:.2f}%", "{:.0f}"],
):
    col.markdown(
        f'<div class="metric-card"><h2>{fmt.format(val)}</h2><p>{label}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 EDA", "🤖 ML Models", "🔍 Anomaly",
    "🔬 Clustering", "📈 Forecasting", "🎨 Advanced",
    "✏️ Data Editor", "💬 Feedback", "🛡️ Admin",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-header">Distribution of Key Variables</p>', unsafe_allow_html=True)
    st.plotly_chart(eda.histogram_grid(df), use_container_width=True)

    st.markdown('<p class="section-header">Time-Series Trends</p>', unsafe_allow_html=True)
    metric = st.selectbox("Metric", ["inflation_rate","interest_rate","gdp_growth","unemployment_rate"], key="ts_metric")
    st.plotly_chart(eda.line_trends(df, selected_countries or all_countries[:6], metric), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">Top Countries by Avg Inflation</p>', unsafe_allow_html=True)
        st.plotly_chart(eda.avg_inflation_bar(df), use_container_width=True)
    with col_b:
        st.markdown('<p class="section-header">Region Composition</p>', unsafe_allow_html=True)
        st.plotly_chart(eda.region_pie(df), use_container_width=True)

    st.markdown('<p class="section-header">Inflation Boxplot by Country</p>', unsafe_allow_html=True)
    st.plotly_chart(eda.boxplot_inflation(df), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<p class="section-header">Violin Plot by Year</p>', unsafe_allow_html=True)
        st.plotly_chart(eda.violin_by_year(df), use_container_width=True)
    with col_d:
        st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
        st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)

    st.markdown('<p class="section-header">Scatter Matrix</p>', unsafe_allow_html=True)
    st.plotly_chart(eda.scatter_matrix(df), use_container_width=True)
    log_action(user, "eda_view", rows=len(df))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML Models
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">Neural Network — Inflation Predictor</p>', unsafe_allow_html=True)
    st.info("3-layer feedforward network trained on the filtered dataset.")

    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner(f"Training for {epochs} epochs…"):
            preds, actual, losses, mse, r2, model_nn, X_t, feat_names = \
                train_and_evaluate(df, epochs=epochs)

        y_t = torch.tensor(actual.reshape(-1, 1), dtype=torch.float32)

        # Confidence score: R² mapped to 0-100%
        confidence = max(0.0, min(1.0, r2)) * 100
        badge_color = "#22c55e" if confidence >= 70 else "#f59e0b" if confidence >= 40 else "#ef4444"
        m1, m2, m3 = st.columns(3)
        m1.metric("MSE", f"{mse:.4f}")
        m2.metric("R² Score", f"{r2:.4f}")
        m3.markdown(
            f'<div style="padding-top:0.6rem">'
            f'<span class="confidence-badge" style="background:{badge_color};color:#fff">'
            f'Model Confidence: {confidence:.1f}%</span></div>',
            unsafe_allow_html=True,
        )

        col_e, col_f = st.columns(2)
        with col_e:
            st.plotly_chart(actual_vs_predicted_plot(actual, preds), use_container_width=True)
        with col_f:
            st.plotly_chart(loss_curve_plot(losses), use_container_width=True)

        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
        st.plotly_chart(permutation_importance(model_nn, X_t, y_t, feat_names), use_container_width=True)

        log_action(user, "model_train", f"epochs={epochs} r2={r2:.3f}", rows=len(df))
        fire("model_trained", {"user": user, "r2": r2, "mse": mse, "epochs": epochs})
    else:
        st.markdown("👆 Click **Train Model** to run the neural network.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-header">Z-Score Anomaly Detection</p>', unsafe_allow_html=True)
    z_thresh = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.5, key="z_thresh")
    df_z, anom_z = anomaly.zscore_anomalies(df, threshold=z_thresh)
    st.plotly_chart(anomaly.zscore_plot(df_z, anom_z, threshold=z_thresh), use_container_width=True)
    pct = len(anom_z) / max(len(df_z), 1) * 100
    st.caption(f"Detected **{len(anom_z)}** anomalies ({pct:.1f}% of data).")
    if len(anom_z) > 0:
        fire("anomaly_found", {"user": user, "method": "zscore", "count": len(anom_z)})

    st.markdown('<p class="section-header">Autoencoder Anomaly Detection</p>', unsafe_allow_html=True)
    ae_pct = st.slider("Anomaly Percentile Threshold", 80, 99, 95, key="ae_pct")

    if st.button("🔍 Run Autoencoder", key="ae_btn"):
        with st.spinner("Training autoencoder…"):
            df_ae, ae_thresh = anomaly.autoencoder_anomalies(df, percentile=ae_pct)
        st.plotly_chart(anomaly.autoencoder_plot(df_ae, ae_thresh), use_container_width=True)
        n_ae = int(df_ae["is_anomaly"].sum())
        st.caption(f"Detected **{n_ae}** anomalies (top {100 - ae_pct}% reconstruction error).")
        log_action(user, "anomaly_autoencoder", f"n={n_ae}", rows=len(df))
        fire("anomaly_found", {"user": user, "method": "autoencoder", "count": n_ae})
    else:
        st.markdown("👆 Click **Run Autoencoder** to detect anomalies.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Clustering
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">Hierarchical Clustering Dendrogram</p>', unsafe_allow_html=True)
    dend_b64 = clustering.dendrogram_figure(df)
    st.image(base64.b64decode(dend_b64), use_container_width=True)

    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown('<p class="section-header">Elbow Method</p>', unsafe_allow_html=True)
        st.plotly_chart(clustering.elbow_plot(df), use_container_width=True)
    with col_h:
        st.markdown(f'<p class="section-header">K-Means (k={n_clusters})</p>', unsafe_allow_html=True)
        st.plotly_chart(clustering.kmeans_scatter(df, k=n_clusters), use_container_width=True)
    log_action(user, "clustering", f"k={n_clusters}", rows=len(df))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Forecasting
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-header">LSTM Inflation Forecasting</p>', unsafe_allow_html=True)
    fc_country = st.selectbox(
        "Select Country", all_countries,
        index=all_countries.index("USA") if "USA" in all_countries else 0,
        key="fc_country",
    )

    if st.button("📈 Run Forecast", key="fc_btn"):
        with st.spinner(f"Training LSTM for {fc_country}…"):
            series, hist_years, future, future_years = forecasting.forecast_country(
                df_raw, fc_country, forecast_years=forecast_years
            )
        if series is not None:
            st.plotly_chart(
                forecasting.forecast_plot(series, hist_years, future, future_years, fc_country),
                use_container_width=True,
            )
            fc_df = pd.DataFrame({
                "Year": future_years,
                "Forecasted Inflation (%)": np.round(future, 3),
                "Lower Bound (−1σ)": np.round(future - np.std(series), 3),
                "Upper Bound (+1σ)": np.round(future + np.std(series), 3),
            })
            st.dataframe(fc_df, use_container_width=True)

            # Confidence score based on historical volatility
            cv = np.std(series) / (np.mean(np.abs(series)) + 1e-8)
            conf = max(0, 100 - cv * 100)
            badge = "#22c55e" if conf >= 70 else "#f59e0b" if conf >= 40 else "#ef4444"
            st.markdown(
                f'<span class="confidence-badge" style="background:{badge};color:#fff">'
                f'Forecast Confidence: {conf:.1f}%</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="citation-box">Forecast generated by a 2-layer LSTM '
                '(hidden=64) trained on World Bank annual CPI data. '
                'Confidence band = ±1 standard deviation of historical series. '
                'Not financial advice.</div>',
                unsafe_allow_html=True,
            )
            log_action(user, "forecast", f"country={fc_country} horizon={forecast_years}", rows=len(series))
            fire("forecast_done", {"user": user, "country": fc_country, "horizon": forecast_years})
        else:
            st.warning("Not enough data for this country. Try another.")
    else:
        st.markdown("👆 Select a country and click **Run Forecast**.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Advanced Plots
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-header">3D Scatter: Oil × Food × Inflation</p>', unsafe_allow_html=True)
    st.plotly_chart(advanced_plots.scatter_3d(df), use_container_width=True)

    col_i, col_j = st.columns(2)
    with col_i:
        st.markdown('<p class="section-header">Contour Density: Oil vs Interest Rate</p>', unsafe_allow_html=True)
        st.plotly_chart(advanced_plots.contour_density(df), use_container_width=True)
    with col_j:
        st.markdown('<p class="section-header">Hexbin: Money Supply vs Inflation</p>', unsafe_allow_html=True)
        hb_b64 = advanced_plots.hexbin_plot(df)
        st.image(base64.b64decode(hb_b64), use_container_width=True)

    st.markdown('<p class="section-header">Facet Grid: Country Inflation Trajectories</p>', unsafe_allow_html=True)
    facet_sel = st.multiselect(
        "Countries for Facet", all_countries,
        default=(selected_countries or all_countries)[:4], key="facet_sel",
    )
    if facet_sel:
        st.plotly_chart(advanced_plots.facet_inflation(df, facet_sel), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Editable Data
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<p class="section-header">✏️ Editable Data Table</p>', unsafe_allow_html=True)
    st.info("Edit values directly. Changes apply to your session only and are not persisted.")

    edit_cols = ["country", "year", "inflation_rate", "interest_rate",
                 "gdp_growth", "unemployment_rate"]
    avail_edit = [c for c in edit_cols if c in df.columns]
    edited_df = st.data_editor(
        df[avail_edit].reset_index(drop=True),
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_bytes = edited_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download as CSV", csv_bytes, "inflation_data.csv", "text/csv")
    with col_dl2:
        json_bytes = edited_df.to_json(orient="records", indent=2).encode()
        st.download_button("⬇️ Download as JSON", json_bytes, "inflation_data.json", "application/json")

    log_action(user, "data_editor_view", rows=len(edited_df))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — Feedback
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<p class="section-header">💬 Feedback Loop</p>', unsafe_allow_html=True)
    st.markdown("Help improve the dashboard — your feedback is logged and reviewed.")

    with st.form("feedback_form"):
        page    = st.selectbox("Which section?",
                               ["EDA", "ML Models", "Anomaly Detection",
                                "Clustering", "Forecasting", "Advanced Plots", "General"])
        rating  = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 4)
        comment = st.text_area("Comments or suggestions")
        if st.form_submit_button("Submit Feedback"):
            submit_feedback(user, page, rating, comment)
            fire("feedback_received", {"user": user, "page": page, "rating": rating})
            st.success("Thanks for your feedback! 🙏")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — Admin
# ─────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    render_admin_panel(user)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#475569;font-size:0.8rem;'>"
    "Global Inflation Insights · "
    "<a href='https://data.worldbank.org' style='color:#38bdf8'>World Bank Open Data</a> · "
    "Streamlit · PyTorch · Plotly"
    "</center>",
    unsafe_allow_html=True,
)
