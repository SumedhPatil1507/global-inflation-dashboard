"""
Macro Research Agent UI — Streamlit interactive module.
Powered by LangGraph, Macro Data Warehouse, Quant Backtest Engine, and World Bank/RBI RAG Knowledge Base.
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure backend can be imported if running directly from inflation-dashboard
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.services.macro_agent import run_macro_research
from backend.services.macro_rag import retrieve_macro_reports
from backend.core.security import create_access_token


def plot_macro_time_series(records: list[dict], country: str) -> go.Figure:
    """Create interactive multi-axis Plotly chart for Inflation, Interest Rate, GDP & Commodities."""
    if not records:
        fig = go.Figure()
        fig.update_layout(title="No Time-Series Data Available", template="plotly_dark")
        return fig

    df_ts = pd.DataFrame(records).sort_values("year")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"📈 {country} — Headline Inflation vs Policy Rate vs GDP Growth (%)",
            "🛢️ Commodity & Macro Proxies",
        ),
        row_heights=[0.65, 0.35],
    )

    # 1. Inflation Rate
    if "inflation_rate" in df_ts.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ts["year"],
                y=df_ts["inflation_rate"],
                name="CPI Inflation (%)",
                mode="lines+markers",
                line=dict(color="#f43f5e", width=3),
                marker=dict(size=8, symbol="circle"),
                hovertemplate="Year: %{x}<br>Inflation: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )

    # 2. Interest Rate
    if "interest_rate" in df_ts.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ts["year"],
                y=df_ts["interest_rate"],
                name="Policy Rate (%)",
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5, dash="dash"),
                marker=dict(size=7, symbol="square"),
                hovertemplate="Year: %{x}<br>Policy Rate: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )

    # 3. GDP Growth
    if "gdp_growth" in df_ts.columns:
        fig.add_trace(
            go.Bar(
                x=df_ts["year"],
                y=df_ts["gdp_growth"],
                name="Real GDP Growth (%)",
                marker_color="rgba(34, 197, 94, 0.45)",
                marker_line=dict(color="#22c55e", width=1.5),
                hovertemplate="Year: %{x}<br>GDP Growth: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )

    # 4. Oil / Gold Proxies
    if "oil_price" in df_ts.columns and not df_ts["oil_price"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df_ts["year"],
                y=df_ts["oil_price"],
                name="Crude Oil ($/bbl)",
                mode="lines+markers",
                line=dict(color="#fb923c", width=2),
                hovertemplate="Year: %{x}<br>Crude Oil: $%{y:.2f}<extra></extra>",
            ),
            row=2, col=1,
        )

    if "gold_price" in df_ts.columns and not df_ts["gold_price"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df_ts["year"],
                y=df_ts["gold_price"],
                name="Gold ($/oz)",
                mode="lines+markers",
                line=dict(color="#facc15", width=2),
                hovertemplate="Year: %{x}<br>Gold: $%{y:.2f}<extra></extra>",
                yaxis="y4" if "oil_price" in df_ts.columns else "y3",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155", dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="#334155")

    return fig


def plot_backtest_charts(backtest_res: dict) -> tuple[go.Figure, go.Figure]:
    """Create interactive Equity Curve and Drawdown plots for quantitative portfolio."""
    dates = backtest_res.get("dates", [])
    equity = backtest_res.get("equity_curve", [])

    if not equity or not dates:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark")
        return empty_fig, empty_fig

    # Equity Curve Plot
    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            name="Portfolio Equity ($10k Base)",
            mode="lines",
            line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            hovertemplate="Date: %{x}<br>Portfolio Value: $%{y:,.2f}<extra></extra>",
        )
    )
    fig_equity.update_layout(
        title="📈 Inflation-Hedge Portfolio Equity Growth ($10,000 Base)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=380,
        margin=dict(l=40, r=40, t=50, b=30),
        hovermode="x unified",
    )
    fig_equity.update_xaxes(showgrid=True, gridcolor="#334155")
    fig_equity.update_yaxes(showgrid=True, gridcolor="#334155", tickprefix="$")

    # Drawdown Plot
    s_eq = pd.Series(equity)
    cummax = s_eq.cummax()
    dd = -100.0 * ((cummax - s_eq) / (cummax + 1e-9))

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dates,
            y=dd.tolist(),
            name="Underwater Drawdown (%)",
            mode="lines",
            line=dict(color="#ef4444", width=2),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.2)",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )
    fig_dd.update_layout(
        title="📉 Underwater Drawdown Profile (%)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=300,
        margin=dict(l=40, r=40, t=50, b=30),
        hovermode="x unified",
    )
    fig_dd.update_xaxes(showgrid=True, gridcolor="#334155")
    fig_dd.update_yaxes(showgrid=True, gridcolor="#334155", ticksuffix="%")

    return fig_equity, fig_dd


def render_macro_research_agent(user: str, role: str):
    """Render the full interactive Macro Research Agent tab."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #38bdf8; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h2 style="color: #38bdf8; margin: 0; font-size: 1.6rem;">🔬 Macro Research Agent (LangGraph + RAG)</h2>
                    <p style="color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                        Autonomous multi-step economic reasoning with tool access to FRED/World Bank data endpoints,
                        Quant Backtester, and embedded World Bank / RBI reports.
                    </p>
                </div>
                <div>
                    <span style="background: #0284c7; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                        RS256 JWT Scoped
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── One-Click Preset Queries ──────────────────────────────────────────────
    st.markdown("##### ⚡ Quick Research Prompts:")
    c1, c2, c3, c4 = st.columns(4)

    preset_query = None
    if c1.button("🇮🇳 Why did inflation move in India this quarter?", key="q_ind"):
        preset_query = "Why did inflation move this quarter for India and what was the RBI MPC stance on food prices?"
    if c2.button("🇺🇸 Why did US CPI move & what was Fed action?", key="q_usa"):
        preset_query = "Why did US inflation move this quarter and how are shelter vs energy costs impacting Fed rate cuts?"
    if c3.button("📊 Quant backtest inflation hedge (Gold, TIPS, Oil)", key="q_hedge"):
        preset_query = "Backtest an inflation-hedging portfolio of GLD, TIP, and USO over recent macro volatility cycles."
    if c4.button("🌐 World Bank Global Disinflation Outlook", key="q_wb"):
        preset_query = "What is the World Bank outlook for global disinflation and emerging market commodity risks?"

    # ── Query Input Form ──────────────────────────────────────────────────────
    with st.form("macro_agent_form"):
        col_q, col_c = st.columns([3, 1])
        with col_q:
            default_text = preset_query or "Why did inflation move this quarter?"
            user_query = st.text_area(
                "Macro Research Question",
                value=default_text,
                height=80,
                help="Ask any question about inflation movements, central bank policy, supply chains, or quant hedges.",
            )
        with col_c:
            agent_country = st.selectbox("Target Economy", ["IND", "USA", "CHN", "GBR", "BRA", "Global"], index=0)
            include_backtest = st.checkbox("Include Quant Backtest", value=True)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            year_range = st.slider("Historical Time Series Horizon", 2018, 2024, (2020, 2024), key="agent_yrs")
        with col_opt2:
            asset_basket = st.multiselect(
                "Backtest Asset Tickers",
                ["GLD", "SPY", "TIP", "USO", "BND", "DBA"],
                default=["GLD", "SPY", "TIP"],
            )

        submit_btn = st.form_submit_button("🚀 Run LangGraph Macro Agent", use_container_width=True)

    # Check if we should execute (either on form submit or when clicking a preset chip)
    if submit_btn or preset_query:
        query_to_run = user_query if submit_btn else preset_query
        target_country = agent_country

        with st.spinner("🤖 LangGraph Agent executing multi-step reasoning workflow…"):
            agent_res = run_macro_research(
                query=query_to_run,
                country=target_country,
                start_year=year_range[0],
                end_year=year_range[1],
                include_backtest=include_backtest,
                backtest_tickers=asset_basket,
            )

        # Store in session state for tab persistence
        st.session_state["macro_agent_last_result"] = agent_res

    # Display results if available
    if "macro_agent_last_result" in st.session_state:
        res = st.session_state["macro_agent_last_result"]

        # ── KPI Cards Summary ─────────────────────────────────────────────────
        ts_data = res.get("time_series_data", {})
        latest_inf = ts_data.get("latest_inflation", 0.0)
        yoy_delta = ts_data.get("yoy_inflation_delta", 0.0)
        pol_rate = ts_data.get("policy_rate", 0.0)
        real_rate = ts_data.get("real_interest_rate", 0.0)

        st.markdown("### 📊 Empirical Macro Snapshot")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.markdown(
            f'<div class="metric-card"><h2>{latest_inf:.2f}%</h2><p>Headline CPI ({ts_data.get("latest_year", 2024)})</p></div>',
            unsafe_allow_html=True,
        )
        delta_color = "#ef4444" if yoy_delta > 0 else "#22c55e"
        kpi2.markdown(
            f'<div class="metric-card"><h2 style="color:{delta_color}">{yoy_delta:+.2f}%</h2><p>Quarterly/YoY CPI Shift</p></div>',
            unsafe_allow_html=True,
        )
        kpi3.markdown(
            f'<div class="metric-card"><h2>{pol_rate:.2f}%</h2><p>Central Bank Policy Rate</p></div>',
            unsafe_allow_html=True,
        )
        real_color = "#22c55e" if real_rate > 0 else "#f59e0b"
        kpi4.markdown(
            f'<div class="metric-card"><h2 style="color:{real_color}">{real_rate:+.2f}%</h2><p>Real Policy Rate Spread</p></div>',
            unsafe_allow_html=True,
        )

        # ── Step Breakdown & Visualizations Tabs ──────────────────────────────
        v_tabs = st.tabs([
            "📑 Research Memorandum & Citations",
            "📈 Interactive Time-Series",
            "📊 Quant Backtest Engine",
            "📚 RAG Document Excerpts",
            "🗺️ LangGraph Execution Trace",
        ])

        # TAB 1: Synthesis & Citations
        with v_tabs[0]:
            st.markdown(res.get("synthesis", "No synthesis generated."))
            st.markdown("---")
            # Export Buttons
            export_col1, export_col2 = st.columns([1, 4])
            with export_col1:
                st.download_button(
                    "⬇️ Export Research Memorandum (MD)",
                    data=res.get("synthesis", "").encode("utf-8"),
                    file_name=f"macro_research_{res.get('country','IND')}.md",
                    mime="text/markdown",
                )

        # TAB 2: Time-Series Plotly Charts
        with v_tabs[1]:
            st.markdown("#### 📈 Interactive Macroeconomic Trends (FRED + World Bank)")
            fig_ts = plot_macro_time_series(ts_data.get("data_points", []), res.get("country", "IND"))
            st.plotly_chart(fig_ts, use_container_width=True)

            # Data Table
            if ts_data.get("data_points"):
                with st.expander("🔍 View Raw Time-Series Data Table"):
                    st.dataframe(pd.DataFrame(ts_data.get("data_points")), use_container_width=True)

        # TAB 3: Quant Backtest Engine
        with v_tabs[2]:
            bt = res.get("backtest_result")
            if bt:
                st.markdown("#### 📊 Quantitative Inflation-Hedge Performance")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Return", f"{bt.get('total_return', 0.0)*100:.2f}%")
                b2.metric("Sharpe Ratio", f"{bt.get('sharpe', 0.0):.2f}")
                b3.metric("Max Drawdown", f"{bt.get('max_drawdown', 0.0)*100:.2f}%")
                b4.metric("Ann. Volatility", f"{bt.get('annualized_volatility', 0.0)*100:.2f}%")

                fig_eq, fig_dd = plot_backtest_charts(bt)
                st.plotly_chart(fig_eq, use_container_width=True)
                st.plotly_chart(fig_dd, use_container_width=True)
                st.caption(f"Weights: {bt.get('ticker_weights', {})}")
            else:
                st.info("No quant backtest requested for this query. Enable 'Include Quant Backtest' to simulate.")

        # TAB 4: RAG Documents
        with v_tabs[3]:
            st.markdown("#### 📚 Retrieved World Bank & RBI Research Documents")
            docs = res.get("rag_documents", [])
            for i, doc in enumerate(docs, 1):
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background:#1e293b; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 1rem; margin-bottom: 0.8rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h4 style="color:#38bdf8; margin:0;">#{i} {doc.get('title')}</h4>
                                <span style="background:#0284c7; color:#fff; font-size:0.75rem; padding:2px 8px; border-radius:12px;">Score: {doc.get('score', 0.0)}</span>
                            </div>
                            <p style="color:#cbd5e1; font-size:0.85rem; margin:0.4rem 0;"><b>Publisher:</b> {doc.get('publisher')} · <b>Citation:</b> <i>{doc.get('citation')}</i> · <b>Date:</b> {doc.get('date')}</p>
                            <p style="color:#94a3b8; font-size:0.9rem; margin-top:0.5rem; line-height:1.5;">{doc.get('content')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # TAB 5: LangGraph Execution Trace
        with v_tabs[4]:
            st.markdown("#### 🗺️ LangGraph Multi-Step Execution Plan")
            for step in res.get("plan_steps", []):
                st.markdown(f"- `{step}`")

            st.markdown("#### ⏱️ Node Execution Logs")
            trace = res.get("execution_trace", [])
            st.dataframe(pd.DataFrame(trace), use_container_width=True)
