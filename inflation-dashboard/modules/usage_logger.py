"""
Usage logs, cost tracking, and feedback loop.
Persists to a local SQLite DB (usage.db).
Swap for Postgres by setting DATABASE_URL in .streamlit/secrets.toml.
"""
import sqlite3
import datetime
import os
import streamlit as st
import pandas as pd

# Use cwd so it works on both local and Streamlit Cloud
DB_PATH = os.path.join(os.getcwd(), "usage.db")

COST_PER_1K = 0.002  # $0.002 per 1k rows processed


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT,
            username  TEXT,
            action    TEXT,
            detail    TEXT,
            tokens    INTEGER DEFAULT 0,
            cost_usd  REAL    DEFAULT 0.0
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT,
            username  TEXT,
            page      TEXT,
            rating    INTEGER,
            comment   TEXT
        )
    """)
    con.commit()
    return con


def log_action(username: str, action: str, detail: str = "", rows: int = 0):
    cost = round((rows / 1000) * COST_PER_1K, 6)
    ts   = datetime.datetime.utcnow().isoformat()
    try:
        con = _conn()
        con.execute(
            "INSERT INTO usage_logs (ts,username,action,detail,tokens,cost_usd) VALUES (?,?,?,?,?,?)",
            (ts, username, action, detail, rows, cost),
        )
        con.commit()
        con.close()
    except Exception:
        pass  # never crash the app over logging


def submit_feedback(username: str, page: str, rating: int, comment: str):
    ts = datetime.datetime.utcnow().isoformat()
    try:
        con = _conn()
        con.execute(
            "INSERT INTO feedback (ts,username,page,rating,comment) VALUES (?,?,?,?,?)",
            (ts, username, page, rating, comment),
        )
        con.commit()
        con.close()
    except Exception:
        pass


def get_usage_df() -> pd.DataFrame:
    try:
        con = _conn()
        df  = pd.read_sql("SELECT * FROM usage_logs ORDER BY id DESC LIMIT 500", con)
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_feedback_df() -> pd.DataFrame:
    try:
        con = _conn()
        df  = pd.read_sql("SELECT * FROM feedback ORDER BY id DESC LIMIT 200", con)
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def render_admin_panel(username: str):
    if username != "admin":
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
            import plotly.express as px
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
