"""
Usage logs, cost tracking, feedback, admin panel.
Delegates persistence to modules/db.py (Supabase or SQLite).
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from modules.db import log_action, submit_feedback, get_usage_df, get_feedback_df

__all__ = ["log_action", "submit_feedback", "render_admin_panel"]


def render_admin_panel(username: str):
    from modules.security import current_role
    if current_role() != "admin":
        st.warning("🔒 Admin access only.")
        return

    st.markdown("## 🛡️ Admin Panel")
    tab_u, tab_f, tab_c = st.tabs(["📋 Usage Logs", "💬 Feedback", "💰 Cost Tracking"])

    with tab_u:
        df_u = get_usage_df()
        if df_u.empty:
            st.info("No usage data yet.")
        else:
            st.dataframe(df_u, use_container_width=True)
            st.download_button("⬇️ Export CSV",
                               df_u.to_csv(index=False).encode(),
                               "usage_logs.csv", "text/csv")

    with tab_f:
        df_f = get_feedback_df()
        if df_f.empty:
            st.info("No feedback yet.")
        else:
            st.metric("Average Rating", f"{df_f['rating'].mean():.1f} / 5")
            st.dataframe(df_f, use_container_width=True)

    with tab_c:
        df_u = get_usage_df()
        if df_u.empty:
            st.info("No usage data yet.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Total Rows Processed", f"{int(df_u['tokens'].sum()):,}")
            c2.metric("Estimated Cost (USD)",  f"${df_u['cost_usd'].sum():.4f}")
            daily = df_u.copy()
            daily["date"] = pd.to_datetime(daily["ts"]).dt.date
            daily_cost    = daily.groupby("date")["cost_usd"].sum().reset_index()
            st.plotly_chart(
                px.bar(daily_cost, x="date", y="cost_usd",
                       title="Daily Cost (USD)", labels={"cost_usd": "Cost ($)"}),
                use_container_width=True,
            )
