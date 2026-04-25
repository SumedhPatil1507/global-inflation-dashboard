
"""
🌍 Global Inflation Insights — Production Dashboard
World Bank + FRED live data · PyTorch ML · Supabase · Role-based auth
"""
# Only import what's needed at startup — torch/scipy/sklearn loaded lazily per tab
import base64
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Global Inflation Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#0f172a; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
.metric-card {
    background:linear-gradient(135deg,#1e293b,#0f172a);
    border:1px solid #334155; border-radius:12px;
    padding:1.2rem 1.5rem; text-align:center; margin-bottom:0.5rem;
}
.metric-card h2 { color:#38bdf8; font-size:2rem; margin:0; }
.metric-card p  { color:#94a3b8; font-size:0.85rem; margin:0; }
.section-header { font-size:1.25rem; font-weight:700; color:#38bdf8;
                  margin-top:1.2rem; margin-bottom:0.4rem; }
.citation-box   { background:#1e293b; border-left:4px solid #38bdf8;
                  padding:0.6rem 1rem; border-radius:6px;
                  font-size:0.8rem; color:#94a3b8; margin-top:0.5rem; }
.confidence-badge { display:inline-block; padding:3px 12px; border-radius:20px;
                    font-size:0.8rem; font-weight:600; }
.data-badge { display:inline-block; padding:2px 8px; border-radius:4px;
              font-size:0.72rem; font-weight:600; background:#1e293b;
              color:#38bdf8; border:1px solid #334155; margin-left:6px; }
</style>
""", unsafe_allow_html=True)

# ── Lightweight imports only at startup ───────────────────────────────────────
from modules.security     import login_wall, logout_button, current_user, current_role, can_access
from modules.db           import log_action, submit_feedback
from modules.usage_logger import render_admin_panel
from modules.webhooks     import fire, render_webhook_settings
from modules.data_loader  import get_data
from modules.insights     import render_insights, render_alert_settings, check_alerts

# Heavy modules — imported lazily inside tabs so startup is instant
def _eda():
    from modules import eda; return eda

def _anomaly():
    from modules import anomaly; return anomaly

def _clustering():
    from modules import clustering; return clustering

def _forecasting():
    from modules import forecasting; return forecasting

def _advanced():
    from modules import advanced_plots; return advanced_plots

def _models():
    from modules.models import (train_and_evaluate, load_saved_model,
                                actual_vs_predicted_plot, loss_curve_plot,
                                permutation_importance)
    return train_and_evaluate, load_saved_model, actual_vs_predicted_plot, \
           loss_curve_plot, permutation_importance

def _export_pdf():
    from modules.insights import export_pdf; return export_pdf

# ── Auth ──────────────────────────────────────────────────────────────────────
login_wall()
user = current_user()
role = current_role()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🌍 Inflation Insights")
    st.markdown(f"👤 **{user}** · `{role}`")
    logout_button()
    st.markdown("---")

    data_source = st.radio("Data Source", ["Live (World Bank + FRED)", "Synthetic Demo"], index=0)
    source_key  = "live" if "Live" in data_source else "synthetic"

    st.markdown("### 📅 Year Range")
    year_range = st.slider("Years", 2015, 2024, (2020, 2024))

    st.markdown("### 🌐 Countries")
    country_ph = st.empty()

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    epochs         = st.slider("Training Epochs",        10, 100, 40, step=10)
    forecast_years = st.slider("Forecast Horizon (yrs)",  1,  10,  5)
    n_clusters     = st.slider("K-Means Clusters",         2,   8,  4)

    st.markdown("---")
    render_alert_settings()

    st.markdown("---")
    render_webhook_settings()
    st.markdown("---")
    st.caption("Global Inflation Insights · World Bank · FRED · PyTorch")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df_raw = get_data(source=source_key, start_year=year_range[0], end_year=year_range[1])

all_countries = sorted(df_raw["country"].unique().tolist())
default_sel   = [c for c in ["USA", "IND", "CHN", "GBR", "BRA"] if c in all_countries]

with st.sidebar:
    selected_countries = country_ph.multiselect(
        "Filter countries", all_countries, default=default_sel or all_countries[:5]
    )

df = df_raw[df_raw["country"].isin(selected_countries)] if selected_countries else df_raw
log_action(user, "data_load", f"source={source_key} rows={len(df)}", rows=len(df))

# Fire threshold alerts
for alert in check_alerts(df):
    fire("threshold_breach", {**alert, "user": user})

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("# 🌍 Global Inflation Insights")
with col_badge:
    src     = df_raw["data_source"].iloc[0] if "data_source" in df_raw.columns else "Live"
    updated = df_raw["last_updated"].iloc[0] if "last_updated" in df_raw.columns else "—"
    st.markdown(f'<br><span class="data-badge">📡 {src}</span>', unsafe_allow_html=True)
    st.caption(f"Updated: {updated}")

st.markdown(
    '<div class="citation-box">📚 <b>Sources:</b> '
    '<a href="https://data.worldbank.org" target="_blank">World Bank Open Data</a> '
    '(CPI, GDP, Unemployment, Lending Rate) · '
    '<a href="https://fred.stlouisfed.org" target="_blank">FRED — St. Louis Fed</a> '
    '(US monthly series). Proxy columns (oil, food, M2, supply chain) are synthetic estimates.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
if not df.empty and "inflation_rate" in df.columns:
    avg_inf  = df["inflation_rate"].mean()
    max_inf  = df["inflation_rate"].max()
    min_inf  = df["inflation_rate"].min()
    n_ctries = df["country"].nunique()
else:
    avg_inf = max_inf = min_inf = n_ctries = 0

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

# ── Tabs (role-gated) ─────────────────────────────────────────────────────────
TAB_LABELS = [
    "📊 EDA", "💡 Insights", "🤖 ML Models", "🔍 Anomaly",
    "🔬 Clustering", "📈 Forecasting", "🎨 Advanced",
    "✏️ Data Editor", "💬 Feedback", "🛡️ Admin",
]
tabs = st.tabs(TAB_LABELS)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    if not can_access("eda"):
        st.warning("Your role does not have access to this section.")
    else:
        eda = _eda()
        st.markdown('<p class="section-header">Distribution of Key Variables</p>', unsafe_allow_html=True)
        st.plotly_chart(eda.histogram_grid(df), use_container_width=True)

        st.markdown('<p class="section-header">Time-Series Trends</p>', unsafe_allow_html=True)
        metric = st.selectbox("Metric", ["inflation_rate", "interest_rate",
                                          "gdp_growth", "unemployment_rate"], key="ts_metric")
        st.plotly_chart(eda.line_trends(df, selected_countries or all_countries[:5], metric),
                        use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-header">Top Countries by Avg Inflation</p>', unsafe_allow_html=True)
            st.plotly_chart(eda.avg_inflation_bar(df), use_container_width=True)
        with c2:
            st.markdown('<p class="section-header">Region Composition</p>', unsafe_allow_html=True)
            st.plotly_chart(eda.region_pie(df), use_container_width=True)

        st.markdown('<p class="section-header">Boxplot by Country</p>', unsafe_allow_html=True)
        st.plotly_chart(eda.boxplot_inflation(df), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<p class="section-header">Violin by Year</p>', unsafe_allow_html=True)
            st.plotly_chart(eda.violin_by_year(df), use_container_width=True)
        with c4:
            st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
            st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)

        st.markdown('<p class="section-header">Scatter Matrix</p>', unsafe_allow_html=True)
        st.plotly_chart(eda.scatter_matrix(df), use_container_width=True)
        log_action(user, "eda_view", rows=len(df))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Insights & Alerts
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">💡 Auto-Generated Business Insights</p>', unsafe_allow_html=True)
    st.caption("Flagged based on Z-score deviation from global mean and real interest rate gaps.")
    render_insights(df)

    st.markdown("---")
    st.markdown('<p class="section-header">📊 Inflation vs Central Bank Target Gap</p>', unsafe_allow_html=True)
    if "inflation_rate" in df.columns and "interest_rate" in df.columns:
        latest = df[df["year"] == df["year"].max()].copy()
        latest["real_rate"] = latest["interest_rate"] - latest["inflation_rate"]
        import plotly.express as px
        fig_gap = px.bar(
            latest.sort_values("real_rate"), x="real_rate", y="country",
            orientation="h", color="real_rate",
            color_continuous_scale="RdYlGn",
            title="Real Interest Rate by Country (Interest Rate − Inflation)",
            labels={"real_rate": "Real Rate (%)", "country": ""},
        )
        fig_gap.update_layout(height=500)
        st.plotly_chart(fig_gap, use_container_width=True)
    log_action(user, "insights_view", rows=len(df))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML Models
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if not can_access("models"):
        st.warning("Upgrade to Analyst or Admin to access ML models.")
    else:
        train_and_evaluate, load_saved_model, actual_vs_predicted_plot, \
            loss_curve_plot, permutation_importance = _models()
        import torch
        st.markdown('<p class="section-header">Neural Network — Inflation Predictor</p>', unsafe_allow_html=True)
        st.info("Trained on 80% of data, evaluated on held-out 20% test set.")

        saved = load_saved_model()
        if saved:
            st.success(f"Saved model found — features: {saved['features']}")

        if st.button("🚀 Train Model", key="train_btn"):
            with st.spinner(f"Training for {epochs} epochs…"):
                result = train_and_evaluate(df, epochs=epochs)

            if result[0] is None:
                st.error("Not enough data to train. Select more countries or a wider year range.")
            else:
                preds, actual, losses, mse, r2, model_nn, X_tv, feat_names = result
                y_tv = torch.tensor(actual.reshape(-1, 1), dtype=torch.float32)

                confidence  = max(0.0, min(1.0, r2)) * 100
                badge_color = "#22c55e" if confidence >= 70 else "#f59e0b" if confidence >= 40 else "#ef4444"

                m1, m2, m3 = st.columns(3)
                m1.metric("Test MSE",  f"{mse:.4f}")
                m2.metric("Test R²",   f"{r2:.4f}")
                m3.markdown(
                    f'<div style="padding-top:0.6rem">'
                    f'<span class="confidence-badge" style="background:{badge_color};color:#fff">'
                    f'Confidence: {confidence:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(actual_vs_predicted_plot(actual, preds), use_container_width=True)
                with c2:
                    st.plotly_chart(loss_curve_plot(losses), use_container_width=True)

                st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
                st.plotly_chart(permutation_importance(model_nn, X_tv, y_tv, feat_names),
                                use_container_width=True)

                log_action(user, "model_train", f"epochs={epochs} r2={r2:.3f}", rows=len(df))
                fire("model_trained", {"user": user, "r2": r2, "mse": mse})
        else:
            st.markdown("👆 Click **Train Model** to run.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    if not can_access("anomaly"):
        st.warning("Upgrade to Analyst or Admin to access anomaly detection.")
    else:
        anomaly = _anomaly()
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
            log_action(user, "anomaly_ae", f"n={n_ae}", rows=len(df))
            fire("anomaly_found", {"user": user, "method": "autoencoder", "count": n_ae})
        else:
            st.markdown("👆 Click **Run Autoencoder** to detect anomalies.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Clustering
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    if not can_access("clustering"):
        st.warning("Your role does not have access to clustering.")
    else:
        clustering = _clustering()
        st.markdown('<p class="section-header">Hierarchical Clustering</p>', unsafe_allow_html=True)
        dend_b64 = clustering.dendrogram_figure(df)
        st.image(base64.b64decode(dend_b64), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-header">Elbow Method</p>', unsafe_allow_html=True)
            st.plotly_chart(clustering.elbow_plot(df), use_container_width=True)
        with c2:
            st.markdown(f'<p class="section-header">K-Means (k={n_clusters})</p>', unsafe_allow_html=True)
            st.plotly_chart(clustering.kmeans_scatter(df, k=n_clusters), use_container_width=True)
        log_action(user, "clustering", f"k={n_clusters}", rows=len(df))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Forecasting
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    if not can_access("forecasting"):
        st.warning("Upgrade to Analyst or Admin to access forecasting.")
    else:
        forecasting = _forecasting()
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
                fc_fig = forecasting.forecast_plot(series, hist_years, future, future_years, fc_country)
                st.plotly_chart(fc_fig, use_container_width=True)

                fc_df = pd.DataFrame({
                    "Year":                    future_years,
                    "Forecasted Inflation (%)":np.round(future, 3),
                    "Lower Bound (−1σ)":       np.round(future - np.std(series), 3),
                    "Upper Bound (+1σ)":        np.round(future + np.std(series), 3),
                })
                st.dataframe(fc_df, use_container_width=True)

                cv   = np.std(series) / (np.mean(np.abs(series)) + 1e-8)
                conf = max(0, 100 - cv * 100)
                badge = "#22c55e" if conf >= 70 else "#f59e0b" if conf >= 40 else "#ef4444"
                st.markdown(
                    f'<span class="confidence-badge" style="background:{badge};color:#fff">'
                    f'Forecast Confidence: {conf:.1f}%</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="citation-box">2-layer LSTM (hidden=64) trained on World Bank '
                    'annual CPI. Confidence band = ±1σ of historical series. Not financial advice.</div>',
                    unsafe_allow_html=True,
                )

                # PDF export
                st.markdown("---")
                if st.button("📄 Export PDF Report", key="pdf_btn"):
                    with st.spinner("Generating PDF…"):
                        export_pdf = _export_pdf()
                        pdf_bytes  = export_pdf(df_raw, fc_country, [fc_fig])
                    st.download_button("⬇️ Download PDF", pdf_bytes,
                                       f"{fc_country}_inflation_report.pdf", "application/pdf")

                log_action(user, "forecast", f"country={fc_country}", rows=len(series))
                fire("forecast_done", {"user": user, "country": fc_country, "horizon": forecast_years})
            else:
                st.warning("Not enough data for this country. Try another.")
        else:
            st.markdown("👆 Select a country and click **Run Forecast**.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Advanced Plots
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    if not can_access("advanced"):
        st.warning("Your role does not have access to advanced plots.")
    else:
        advanced_plots = _advanced()
        st.markdown('<p class="section-header">3D: Oil × Food × Inflation</p>', unsafe_allow_html=True)
        st.plotly_chart(advanced_plots.scatter_3d(df), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-header">Contour: Oil vs Interest Rate</p>', unsafe_allow_html=True)
            st.plotly_chart(advanced_plots.contour_density(df), use_container_width=True)
        with c2:
            st.markdown('<p class="section-header">Hexbin: Money Supply vs Inflation</p>', unsafe_allow_html=True)
            hb_b64 = advanced_plots.hexbin_plot(df)
            st.image(base64.b64decode(hb_b64), use_container_width=True)

        st.markdown('<p class="section-header">Facet Grid: Country Trajectories</p>', unsafe_allow_html=True)
        facet_sel = st.multiselect("Countries", all_countries,
                                   default=(selected_countries or all_countries)[:4], key="facet_sel")
        if facet_sel:
            st.plotly_chart(advanced_plots.facet_inflation(df, facet_sel), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Data Editor
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    if not can_access("editor"):
        st.warning("Upgrade to Analyst or Admin to edit data.")
    else:
        st.markdown('<p class="section-header">✏️ Editable Data Table</p>', unsafe_allow_html=True)
        st.info("Session-only edits — not persisted to database.")
        edit_cols  = ["country", "year", "inflation_rate", "interest_rate",
                      "gdp_growth", "unemployment_rate"]
        avail_edit = [c for c in edit_cols if c in df.columns]
        edited_df  = st.data_editor(df[avail_edit].reset_index(drop=True),
                                    num_rows="dynamic", use_container_width=True,
                                    key="data_editor")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ CSV", edited_df.to_csv(index=False).encode(),
                               "inflation_data.csv", "text/csv")
        with c2:
            st.download_button("⬇️ JSON",
                               edited_df.to_json(orient="records", indent=2).encode(),
                               "inflation_data.json", "application/json")
        log_action(user, "data_editor", rows=len(edited_df))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — Feedback
# ─────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    if not can_access("feedback"):
        st.warning("Your role does not have access to feedback.")
    else:
        st.markdown('<p class="section-header">💬 Feedback</p>', unsafe_allow_html=True)
        with st.form("feedback_form"):
            page    = st.selectbox("Section", ["EDA", "Insights", "ML Models", "Anomaly",
                                               "Clustering", "Forecasting", "Advanced", "General"])
            rating  = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 4)
            comment = st.text_area("Comments or suggestions")
            if st.form_submit_button("Submit"):
                submit_feedback(user, page, rating, comment)
                fire("feedback_received", {"user": user, "page": page, "rating": rating})
                st.success("Thanks for your feedback!")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — Admin
# ─────────────────────────────────────────────────────────────────────────────
with tabs[9]:
    render_admin_panel(user)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#475569;font-size:0.8rem'>"
    "Global Inflation Insights · "
    "<a href='https://data.worldbank.org' style='color:#38bdf8'>World Bank</a> · "
    "<a href='https://fred.stlouisfed.org' style='color:#38bdf8'>FRED</a> · "
    "Streamlit · PyTorch · Supabase"
    "</center>",
    unsafe_allow_html=True,
)
