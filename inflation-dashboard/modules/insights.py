"""
Business insight callouts, alert thresholds, and PDF report export.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ── Insight callouts ──────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame) -> list[dict]:
    """
    Returns a list of auto-generated insight dicts:
    {level: 'critical'|'warning'|'info', country, message, value}
    """
    insights = []
    if df.empty or "inflation_rate" not in df.columns:
        return insights

    latest_year = df["year"].max()
    latest      = df[df["year"] == latest_year]

    global_mean = df["inflation_rate"].mean()
    global_std  = df["inflation_rate"].std()

    for _, row in latest.iterrows():
        inf  = row["inflation_rate"]
        z    = (inf - global_mean) / (global_std + 1e-8)
        country = row["country"]

        if z > 3:
            insights.append({"level": "critical", "country": country,
                "value": inf, "z": z,
                "message": f"{country} inflation {inf:.1f}% is {z:.1f}σ above global mean — extreme risk"})
        elif z > 2:
            insights.append({"level": "warning", "country": country,
                "value": inf, "z": z,
                "message": f"{country} inflation {inf:.1f}% is elevated ({z:.1f}σ above mean)"})
        elif z < -1.5:
            insights.append({"level": "info", "country": country,
                "value": inf, "z": z,
                "message": f"{country} inflation {inf:.1f}% is below global mean — deflationary pressure"})

    # Interest rate vs inflation gap
    if "interest_rate" in df.columns:
        for _, row in latest.iterrows():
            gap = row.get("interest_rate", np.nan) - row.get("inflation_rate", np.nan)
            if pd.notna(gap) and gap < -2:
                insights.append({"level": "warning", "country": row["country"],
                    "value": gap,
                    "message": f"{row['country']} real interest rate {gap:.1f}% — policy likely too loose"})

    insights.sort(key=lambda x: {"critical": 0, "warning": 1, "info": 2}[x["level"]])
    return insights[:10]  # top 10


def render_insights(df: pd.DataFrame):
    insights = generate_insights(df)
    if not insights:
        st.info("No significant insights detected for current selection.")
        return

    colors = {"critical": "#ef4444", "warning": "#f59e0b", "info": "#38bdf8"}
    icons  = {"critical": "🔴", "warning": "🟡", "info": "🔵"}

    for ins in insights:
        c = colors[ins["level"]]
        i = icons[ins["level"]]
        st.markdown(
            f'<div style="background:#1e293b;border-left:4px solid {c};'
            f'padding:0.6rem 1rem;border-radius:6px;margin-bottom:0.5rem">'
            f'{i} <b style="color:{c}">{ins["level"].upper()}</b> — '
            f'<span style="color:#e2e8f0">{ins["message"]}</span></div>',
            unsafe_allow_html=True,
        )


# ── Alert thresholds ──────────────────────────────────────────────────────────
def render_alert_settings():
    st.markdown("### 🔔 Alert Thresholds")
    st.caption("Webhooks fire automatically when thresholds are breached.")

    col1, col2 = st.columns(2)
    with col1:
        thresh_inf = st.number_input("Inflation alert above (%)", value=8.0, step=0.5)
    with col2:
        thresh_unemp = st.number_input("Unemployment alert above (%)", value=15.0, step=0.5)

    st.session_state["alert_inflation"]    = thresh_inf
    st.session_state["alert_unemployment"] = thresh_unemp


def check_alerts(df: pd.DataFrame) -> list[dict]:
    """Returns list of breached alerts to fire as webhooks."""
    alerts   = []
    thresh_i = st.session_state.get("alert_inflation", 8.0)
    thresh_u = st.session_state.get("alert_unemployment", 15.0)

    if df.empty:
        return alerts

    latest = df[df["year"] == df["year"].max()]
    for _, row in latest.iterrows():
        if row.get("inflation_rate", 0) > thresh_i:
            alerts.append({"event": "threshold_breach", "country": row["country"],
                           "metric": "inflation_rate", "value": row["inflation_rate"],
                           "threshold": thresh_i})
        if row.get("unemployment_rate", 0) > thresh_u:
            alerts.append({"event": "threshold_breach", "country": row["country"],
                           "metric": "unemployment_rate", "value": row["unemployment_rate"],
                           "threshold": thresh_u})
    return alerts


# ── PDF export ────────────────────────────────────────────────────────────────
def export_pdf(df: pd.DataFrame, country: str, figures: list[go.Figure]) -> bytes:
    """Generate a simple PDF report for a country. Returns bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib import colors

        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   rightMargin=40, leftMargin=40,
                                   topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        story.append(Paragraph(f"Global Inflation Insights — {country}", styles["Title"]))
        story.append(Paragraph(f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                                styles["Normal"]))
        story.append(Spacer(1, 12))

        # Summary table
        sub = df[df["country"] == country].sort_values("year")
        if not sub.empty:
            cols_show = ["year", "inflation_rate", "interest_rate", "gdp_growth", "unemployment_rate"]
            cols_show = [c for c in cols_show if c in sub.columns]
            data = [cols_show] + sub[cols_show].round(2).values.tolist()
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#1e293b"), colors.HexColor("#0f172a")]),
                ("TEXTCOLOR",  (0, 1), (-1, -1), colors.white),
                ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#334155")),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

        # Charts — kaleido optional, skip if not available
        for fig in figures[:3]:
            try:
                img_bytes = fig.to_image(format="png", width=700, height=350, scale=1.5)
                img_buf   = io.BytesIO(img_bytes)
                story.append(Image(img_buf, width=480, height=240))
                story.append(Spacer(1, 8))
            except Exception:
                story.append(Paragraph("[Chart unavailable — install kaleido for chart export]",
                                       styles["Normal"]))
                story.append(Spacer(1, 8))

        # Insights
        story.append(Paragraph("Key Insights", styles["Heading2"]))
        for ins in generate_insights(df):
            story.append(Paragraph(f"• {ins['message']}", styles["Normal"]))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        # Return a minimal error PDF
        buf = io.BytesIO()
        buf.write(f"PDF generation failed: {e}".encode())
        buf.seek(0)
        return buf.read()
