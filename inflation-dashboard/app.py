"""
Global Inflation Insights — Production Dashboard
World Bank + FRED · PyTorch/sklearn ML · Supabase · Role-based auth
"""
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

st.markdown("""<style>
[data-testid="stSidebar"]{background:#0f172a}
[data-testid="stSidebar"] *{color:#e2e8f0 !important}
.metric-card{background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;
  border-radius:12px;padding:1.2rem 1.5rem;text-align:center;margin-bottom:.5rem}
.metric-card h2{color:#38bdf8;font-size:2rem;margin:0}
.metric-card p{color:#94a3b8;font-size:.85rem;margin:0}
.sh{font-size:1.2rem;font-weight:700;color:#38bdf8;margin-top:1rem;margin-bottom:.3rem}
.cb{background:#1e293b;border-left:4px solid #38bdf8;padding:.6rem 1rem;
  border-radius:6px;font-size:.8rem;color:#94a3b8;margin-top:.5rem}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.8rem;font-weight:600}
</style>""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from modules.security     import login_wall, logout_button, current_user, current_role, can_access
from modules.db           import log_action, submit_feedback
from modules.usage_logger import render_admin_panel
from modules.webhooks     import fire, render_webhook_settings
from modules.data_loader  import get_data
from modules.insights     import render_insights, render_alert_settings, check_alerts

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
    data_source = st.radio("Data Source", ["Live (World Bank + FRED)", "Synthetic Demo"], index=1)
    source_key  = "live" if "Live" in data_source else "synthetic"
    st.markdown("### 📅 Year Range")
    year_range = st.slider("Years", 2015, 2024, (2020, 2024))
    st.markdown("### 🌐 Countries")
    country_ph = st.empty()
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    epochs         = st.slider("Training Epochs",       10, 100, 30, step=10)
    forecast_years = st.slider("Forecast Horizon (yrs)", 1,  10,  5)
    n_clusters     = st.slider("K-Means Clusters",        2,   8,  4)
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
default_sel   = [c for c in ["USA","IND","CHN","GBR","BRA"] if c in all_countries]
with st.sidebar:
    selected_countries = country_ph.multiselect(
        "Filter countries", all_countries, default=default_sel or all_countries[:5])

df = df_raw[df_raw["country"].isin(selected_countries)] if selected_countries else df_raw
log_action(user, "data_load", f"source={source_key} rows={len(df)}", rows=len(df))

for alert in check_alerts(df):
    fire("threshold_breach", {**alert, "user": user})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Global Inflation Insights")
src     = df_raw["data_source"].iloc[0] if "data_source" in df_raw.columns else "Synthetic"
updated = df_raw["last_updated"].iloc[0] if "last_updated" in df_raw.columns else "—"
st.markdown(f'<div class="cb">📚 <b>Source:</b> {src} · Updated: {updated} · '
            '<a href="https://data.worldbank.org" target="_blank">World Bank</a> · '
            '<a href="https://fred.stlouisfed.org" target="_blank">FRED</a></div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
avg_inf  = df["inflation_rate"].mean() if (not df.empty and "inflation_rate" in df.columns) else 0
max_inf  = df["inflation_rate"].max()  if (not df.empty and "inflation_rate" in df.columns) else 0
min_inf  = df["inflation_rate"].min()  if (not df.empty and "inflation_rate" in df.columns) else 0
n_ctries = df["country"].nunique()     if (not df.empty and "country" in df.columns) else 0

for col, label, val, fmt in zip(
    st.columns(4),
    ["Avg Inflation","Peak Inflation","Lowest Inflation","Countries"],
    [avg_inf, max_inf, min_inf, n_ctries],
    ["{:.2f}%","{:.2f}%","{:.2f}%","{:.0f}"],
):
    col.markdown(f'<div class="metric-card"><h2>{fmt.format(val)}</h2><p>{label}</p></div>',
                 unsafe_allow_html=True)
st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 EDA", "💡 Insights", "🔬 Macro Agent", "📈 Signals",
    "🤖 ML Models", "🔍 Anomaly", "🔬 Clustering",
    "🔮 Forecasting", "💥 Stress Test",
    "🎨 Advanced", "✏️ Data Editor", "💬 Feedback", "🛡️ Admin",
])

# ── TAB 0: EDA ────────────────────────────────────────────────────────────────
with tabs[0]:
    if not can_access("eda"):
        st.warning("Access restricted.")
    else:
        from modules import eda as _eda_mod
        diag_tab, dist_tab, corr_tab = st.tabs(
            ["📋 Data Diagnostics", "📊 Distributions", "🔗 Correlations"])

        with diag_tab:
            st.markdown('<p class="sh">Boxplots by Country</p>', unsafe_allow_html=True)
            st.plotly_chart(_eda_mod.boxplot_inflation(df), use_container_width=True)
            st.markdown('<p class="sh">Violin by Year</p>', unsafe_allow_html=True)
            st.plotly_chart(_eda_mod.violin_by_year(df), use_container_width=True)

        with dist_tab:
            st.markdown('<p class="sh">Histograms</p>', unsafe_allow_html=True)
            st.plotly_chart(_eda_mod.histogram_grid(df), use_container_width=True)
            st.markdown('<p class="sh">Time-Series Trends</p>', unsafe_allow_html=True)
            metric = st.selectbox("Metric", ["inflation_rate","interest_rate",
                                              "gdp_growth","unemployment_rate"], key="ts_m")
            st.plotly_chart(_eda_mod.line_trends(df, selected_countries or all_countries[:5], metric),
                            use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(_eda_mod.avg_inflation_bar(df), use_container_width=True)
            with c2:
                st.plotly_chart(_eda_mod.region_pie(df), use_container_width=True)

        with corr_tab:
            st.plotly_chart(_eda_mod.correlation_heatmap(df), use_container_width=True)
            st.plotly_chart(_eda_mod.scatter_matrix(df), use_container_width=True)

        log_action(user, "eda_view", rows=len(df))

# ── TAB 1: Insights ───────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="sh">Auto-Generated Business Insights</p>', unsafe_allow_html=True)
    render_insights(df)
    st.markdown("---")
    if "inflation_rate" in df.columns and "interest_rate" in df.columns:
        import plotly.express as px
        latest = df[df["year"] == df["year"].max()].copy()
        latest["real_rate"] = latest["interest_rate"] - latest["inflation_rate"]
        fig_gap = px.bar(latest.sort_values("real_rate"), x="real_rate", y="country",
                         orientation="h", color="real_rate", color_continuous_scale="RdYlGn",
                         title="Real Interest Rate by Country",
                         labels={"real_rate":"Real Rate (%)","country":""})
        fig_gap.update_layout(height=500)
        st.plotly_chart(fig_gap, use_container_width=True)
    log_action(user, "insights_view", rows=len(df))

# ── TAB 2: Macro Research Agent ───────────────────────────────────────────────
with tabs[2]:
    if not can_access("macro_agent"):
        st.warning("Access restricted.")
    else:
        from modules.agent_ui import render_macro_research_agent
        render_macro_research_agent(user, role)
        log_action(user, "macro_agent_view", rows=len(df))

# ── TAB 3: Trading Signals ────────────────────────────────────────────────────
with tabs[3]:
    if not can_access("models"):
        st.warning("Upgrade to Analyst or Admin.")
    else:
        from modules.trading_signals import (yield_optimizer, yield_optimizer_chart,
                                              signal_table, detect_regime, regime_allocation_chart)
        st.markdown('<p class="sh">Inflation-Adjusted Yield Optimizer (Carry Trade Signals)</p>',
                    unsafe_allow_html=True)
        df_yield = yield_optimizer(df)
        if not df_yield.empty:
            st.plotly_chart(yield_optimizer_chart(df_yield), use_container_width=True)
            st.dataframe(df_yield.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="sh">Regime Switching Allocator</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: inf_in  = st.number_input("Inflation (%)",    value=float(round(avg_inf,1)), step=0.5)
        with c2: gdp_in  = st.number_input("GDP Growth (%)",   value=2.0, step=0.5)
        with c3: unemp_in= st.number_input("Unemployment (%)", value=5.0, step=0.5)
        result = detect_regime(inf_in, gdp_in, unemp_in)
        st.markdown(f'<div class="cb">{result["signal"]}</div>', unsafe_allow_html=True)
        st.plotly_chart(regime_allocation_chart(result["allocation"], result["regime"]),
                        use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="sh">Country Signal Table</p>', unsafe_allow_html=True)
        df_sig = signal_table(df)
        if not df_sig.empty:
            st.dataframe(df_sig.reset_index(drop=True), use_container_width=True)
        log_action(user, "signals_view", rows=len(df))

# ── TAB 4: ML Models ─────────────────────────────────────────────────────────
with tabs[4]:
    if not can_access("models"):
        st.warning("Upgrade to Analyst or Admin.")
    else:
        from modules.models import (train_and_evaluate, load_saved_model,
                                    actual_vs_predicted_plot, loss_curve_plot,
                                    permutation_importance, TORCH_OK)
        st.markdown('<p class="sh">Neural Network / Ridge Regression — Inflation Predictor</p>',
                    unsafe_allow_html=True)
        engine = "PyTorch NN" if TORCH_OK else "sklearn Ridge (torch not available)"
        st.info(f"Engine: **{engine}** · Evaluated on held-out 20% test set.")

        saved = load_saved_model()
        if saved:
            st.success(f"Saved model found — features: {saved.get('features','—')}")

        if st.button("🚀 Train Model", key="train_btn"):
            with st.spinner(f"Training for {epochs} epochs…"):
                result = train_and_evaluate(df, epochs=epochs)
            if result[0] is None:
                st.error("Not enough data. Select more countries or a wider year range.")
            else:
                preds, actual, losses, mse, r2, model_obj, X_tv, feat_names = result
                conf  = max(0.0, min(1.0, r2)) * 100
                color = "#22c55e" if conf>=70 else "#f59e0b" if conf>=40 else "#ef4444"
                m1,m2,m3 = st.columns(3)
                m1.metric("Test MSE", f"{mse:.4f}")
                m2.metric("Test R²",  f"{r2:.4f}")
                m3.markdown(f'<div style="padding-top:.6rem"><span class="badge" '
                            f'style="background:{color};color:#fff">Confidence: {conf:.1f}%</span></div>',
                            unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(actual_vs_predicted_plot(actual, preds), use_container_width=True)
                with c2: st.plotly_chart(loss_curve_plot(losses), use_container_width=True)
                if TORCH_OK:
                    try:
                        import torch
                        y_tv = torch.tensor(actual.reshape(-1,1), dtype=torch.float32)
                        st.plotly_chart(permutation_importance(model_obj, X_tv, y_tv, feat_names),
                                        use_container_width=True)
                    except (ImportError, OSError):
                        st.plotly_chart(permutation_importance(model_obj, X_tv, actual, feat_names),
                                        use_container_width=True)
                else:
                    st.plotly_chart(permutation_importance(model_obj, X_tv, actual, feat_names),
                                    use_container_width=True)
                log_action(user, "model_train", f"r2={r2:.3f}", rows=len(df))
                fire("model_trained", {"user": user, "r2": r2, "mse": mse})
        else:
            st.markdown("👆 Click **Train Model** to run.")

# ── TAB 5: Anomaly ────────────────────────────────────────────────────────────
with tabs[5]:
    if not can_access("anomaly"):
        st.warning("Upgrade to Analyst or Admin.")
    else:
        from modules import anomaly as _anom
        st.markdown('<p class="sh">Z-Score Anomaly Detection</p>', unsafe_allow_html=True)
        z_thresh = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.5, key="z_thresh")
        df_z, anom_z = _anom.zscore_anomalies(df, threshold=z_thresh)
        st.plotly_chart(_anom.zscore_plot(df_z, anom_z, threshold=z_thresh), use_container_width=True)
        st.caption(f"Detected **{len(anom_z)}** anomalies ({len(anom_z)/max(len(df_z),1)*100:.1f}%).")
        if len(anom_z) > 0:
            fire("anomaly_found", {"user": user, "method": "zscore", "count": len(anom_z)})

        st.markdown('<p class="sh">Autoencoder Anomaly Detection</p>', unsafe_allow_html=True)
        ae_pct = st.slider("Anomaly Percentile", 80, 99, 95, key="ae_pct")
        if st.button("🔍 Run Autoencoder", key="ae_btn"):
            with st.spinner("Training Autoencoder…"):
                df_ae, anom_ae = _anom.autoencoder_anomalies(df, contamination=1-ae_pct/100)
            if df_ae is not None:
                st.plotly_chart(_anom.autoencoder_loss_plot(df_ae, anom_ae), use_container_width=True)
                st.dataframe(anom_ae[["country","year","inflation_rate","reconstruction_error"]],
                             use_container_width=True)
                log_action(user, "anomaly_autoencoder", f"found={len(anom_ae)}", rows=len(df))
        log_action(user, "anomaly_view", rows=len(df))

# ── TAB 6: Clustering ─────────────────────────────────────────────────────────
with tabs[6]:
    if not can_access("clustering"):
        st.warning("Access restricted.")
    else:
        from modules import clustering as _clust
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<p class="sh">Clustering Controls</p>', unsafe_allow_html=True)
            method = st.radio("Method", ["K-Means", "DBSCAN", "PCA Projection"], key="clust_m")
        with c2:
            if method == "K-Means":
                df_c, sil = _clust.kmeans_clustering(df, n_clusters)
                st.plotly_chart(_clust.cluster_scatter(df_c, "cluster"), use_container_width=True)
                st.metric("Silhouette Score", f"{sil:.3f}")
            elif method == "DBSCAN":
                df_c = _clust.dbscan_clustering(df)
                st.plotly_chart(_clust.cluster_scatter(df_c, "cluster_dbscan"), use_container_width=True)
            else:
                pca_df, var = _clust.pca_reduction(df)
                st.plotly_chart(_clust.pca_scatter(pca_df), use_container_width=True)
                st.caption(f"Explained variance: PC1={var[0]:.1%}, PC2={var[1]:.1%}")
        log_action(user, "clustering_view", f"method={method}", rows=len(df))

# ── TAB 7: Forecasting ────────────────────────────────────────────────────────
with tabs[7]:
    if not can_access("forecasting"):
        st.warning("Upgrade to Analyst or Admin.")
    else:
        from modules.forecasting import (run_arima_forecast, run_prophet_forecast,
                                          run_var_forecast, forecast_plot_dataframe,
                                          var_multi_country_plot, PROPHET_OK)
        st.markdown('<p class="sh">Macroeconomic Time-Series Forecasting</p>', unsafe_allow_html=True)
        fc_country = st.selectbox("Country for Single-Series Forecast",
                                  selected_countries or all_countries, key="fc_c")
        fc_col1, fc_col2 = st.columns(2)
        with fc_col1:
            st.markdown("#### ARIMA Model")
            df_sub = df[df["country"] == fc_country].sort_values("year")
            if len(df_sub) >= 5:
                df_arima = run_arima_forecast(df_sub, steps=forecast_years)
                st.plotly_chart(forecast_plot_dataframe(df_sub, df_arima, fc_country, "ARIMA"),
                                use_container_width=True)
            else:
                st.warning("Need at least 5 observations for ARIMA.")

        with fc_col2:
            st.markdown("#### Prophet / Polynomial Model")
            if len(df_sub) >= 5:
                df_prophet = run_prophet_forecast(df_sub, steps=forecast_years)
                tag = "Prophet" if PROPHET_OK else "Poly-Trend (Prophet fallback)"
                st.plotly_chart(forecast_plot_dataframe(df_sub, df_prophet, fc_country, tag),
                                use_container_width=True)
            else:
                st.warning("Need at least 5 observations.")

        st.markdown("---")
        st.markdown('<p class="sh">Vector Autoregression (VAR) Multi-Country Forecast</p>',
                    unsafe_allow_html=True)
        var_countries = (selected_countries or all_countries)[:4]
        if len(var_countries) >= 2:
            df_var_hist, df_var_fc = run_var_forecast(df, var_countries, steps=forecast_years)
            if not df_var_fc.empty:
                st.plotly_chart(var_multi_country_plot(df_var_hist, df_var_fc, var_countries),
                                use_container_width=True)
        else:
            st.info("Select at least 2 countries to run VAR.")
        log_action(user, "forecasting_view", f"country={fc_country}", rows=len(df))

# ── TAB 8: Stress Test ────────────────────────────────────────────────────────
with tabs[8]:
    if not can_access("models"):
        st.warning("Upgrade to Analyst or Admin.")
    else:
        from modules.forecasting import (run_stress_test, stress_test_plot,
                                          portfolio_bar, SCENARIOS, ASSETS, cumulative_wealth_plot,
                                          sharpe_table, monte_carlo_plot)
        st.markdown('<p class="sh">Portfolio Stress Tester — Macro Scenario Simulator</p>',
                    unsafe_allow_html=True)
        st.caption("Simulates real asset returns under inflation/unemployment shocks over 5–10 years.")

        c1, c2 = st.columns(2)
        with c1:
            scenario = st.selectbox("Scenario", list(SCENARIOS.keys()), key="scenario")
            horizon  = st.slider("Horizon (years)", 3, 10, 7, key="stress_horizon")
        with c2:
            base_inf = st.number_input("Base Inflation (%)", value=float(round(avg_inf,1)), step=0.5)
            if scenario == "Custom":
                c_inf  = st.number_input("Custom Inflation Shock (%)", value=5.0, step=0.5)
                c_unemp= st.number_input("Custom Unemployment Shock (%)", value=3.0, step=0.5)
            else:
                c_inf = c_unemp = 0.0

        st.markdown("**Portfolio Weights** (must sum to 1.0)")
        weights = {}
        cols = st.columns(len(ASSETS))
        for i, asset in enumerate(ASSETS):
            default_w = round(1 / len(ASSETS), 2)
            weights[asset] = cols[i].number_input(asset, 0.0, 1.0, default_w, 0.05,
                                                   key=f"w_{asset}", label_visibility="visible")
        total_w = sum(weights.values())
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.2f} — normalizing automatically.")
            weights = {k: v/total_w for k, v in weights.items()}

        if st.button("💥 Run Stress Test", key="stress_btn"):
            df_stress = run_stress_test(base_inf, scenario, c_inf, c_unemp, horizon, weights)
            st.plotly_chart(stress_test_plot(df_stress, scenario), use_container_width=True)
            st.plotly_chart(portfolio_bar(df_stress), use_container_width=True)

            st.markdown('<p class="sh">Cumulative Wealth ($1 Invested)</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(cumulative_wealth_plot(df_stress, scenario), use_container_width=True)

            st.markdown('<p class="sh">Risk-Adjusted Performance (Sharpe Ratio)</p>',
                        unsafe_allow_html=True)
            df_sharpe = sharpe_table(df_stress)
            st.dataframe(df_sharpe, use_container_width=True)

            st.markdown('<p class="sh">Monte Carlo Simulation</p>', unsafe_allow_html=True)
            best_asset = df_sharpe.iloc[0]["Asset"]
            best_ret   = df_stress[df_stress["Asset"]==best_asset]["Real Return (%)"].mean()
            best_vol   = df_stress[df_stress["Asset"]==best_asset]["Real Return (%)"].std()
            st.plotly_chart(monte_carlo_plot(best_ret, best_vol, horizon,
                                             simulations=300, asset_name=best_asset),
                            use_container_width=True)

            worst = df_sharpe.iloc[-1]["Asset"]
            best  = df_sharpe.iloc[0]["Asset"]
            st.markdown(f'<div class="cb">🔴 Worst risk-adjusted: <b>{worst}</b> '
                        f'(Sharpe: {df_sharpe.iloc[-1]["Sharpe Ratio"]}) · '
                        f'🟢 Best risk-adjusted: <b>{best}</b> '
                        f'(Sharpe: {df_sharpe.iloc[0]["Sharpe Ratio"]})</div>',
                        unsafe_allow_html=True)
            log_action(user, "stress_test", f"scenario={scenario}", rows=horizon)
        else:
            st.markdown("👆 Configure scenario and click **Run Stress Test**.")

# ── TAB 9: Advanced ───────────────────────────────────────────────────────────
with tabs[9]:
    if not can_access("advanced"):
        st.warning("Access restricted.")
    else:
        from modules import advanced_plots as _adv
        st.plotly_chart(_adv.scatter_3d(df), use_container_width=True)
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(_adv.contour_density(df), use_container_width=True)
        with c2: st.plotly_chart(_adv.hexbin_plot(df), use_container_width=True)
        facet_sel = st.multiselect("Countries for Facet", all_countries,
                                   default=(selected_countries or all_countries)[:4], key="facet_sel")
        if facet_sel:
            st.plotly_chart(_adv.facet_inflation(df, facet_sel), use_container_width=True)

# ── TAB 10: Data Editor ───────────────────────────────────────────────────────
with tabs[10]:
    if not can_access("editor"):
        st.warning("Upgrade to Analyst or Admin.")
    else:
        st.info("Session-only edits — not persisted.")
        edit_cols  = [c for c in ["country","year","inflation_rate","interest_rate",
                                   "gdp_growth","unemployment_rate"] if c in df.columns]
        edited_df  = st.data_editor(df[edit_cols].reset_index(drop=True),
                                    num_rows="dynamic", use_container_width=True, key="de")
        c1,c2 = st.columns(2)
        with c1: st.download_button("⬇️ CSV",  edited_df.to_csv(index=False).encode(),
                                    "data.csv","text/csv")
        with c2: st.download_button("⬇️ JSON", edited_df.to_json(orient="records",indent=2).encode(),
                                    "data.json","application/json")

# ── TAB 11: Feedback ──────────────────────────────────────────────────────────
with tabs[11]:
    if not can_access("feedback"):
        st.warning("Access restricted.")
    else:
        with st.form("fb"):
            page    = st.selectbox("Section", ["EDA","Insights","Signals","ML Models",
                                               "Anomaly","Clustering","Forecasting",
                                               "Stress Test","Advanced","General"])
            rating  = st.slider("Rating (1=poor, 5=excellent)", 1, 5, 4)
            comment = st.text_area("Comments")
            if st.form_submit_button("Submit"):
                submit_feedback(user, page, rating, comment)
                fire("feedback_received", {"user": user, "page": page, "rating": rating})
                st.success("Thanks!")

# ── TAB 12: Admin ─────────────────────────────────────────────────────────────
with tabs[12]:
    render_admin_panel(user)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<center style='color:#475569;font-size:.8rem'>Global Inflation Insights · "
            "<a href='https://data.worldbank.org' style='color:#38bdf8'>World Bank</a> · "
            "<a href='https://fred.stlouisfed.org' style='color:#38bdf8'>FRED</a> · "
            "Streamlit · PyTorch · Supabase</center>", unsafe_allow_html=True)
