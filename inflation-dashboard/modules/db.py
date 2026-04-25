"""
Database layer — Supabase (production) with SQLite fallback (local dev).
Set SUPABASE_URL and SUPABASE_KEY in .streamlit/secrets.toml to use Supabase.
"""
import os
import datetime
import sqlite3
import streamlit as st
import pandas as pd

# ── Supabase client (lazy init) ───────────────────────────────────────────────
def _supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def _use_supabase() -> bool:
    try:
        return bool(st.secrets.get("supabase", {}).get("url"))
    except Exception:
        return False


# ── SQLite fallback ───────────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.getcwd(), "usage.db")


def _sqlite_conn():
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS usage_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, username TEXT,
        action TEXT, detail TEXT, tokens INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0.0)""")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, username TEXT,
        page TEXT, rating INTEGER, comment TEXT)""")
    con.commit()
    return con


# ── Public API ────────────────────────────────────────────────────────────────
COST_PER_1K = 0.002


def log_action(username: str, action: str, detail: str = "", rows: int = 0):
    ts   = datetime.datetime.utcnow().isoformat()
    cost = round((rows / 1000) * COST_PER_1K, 6)
    try:
        if _use_supabase():
            _supabase().table("usage_logs").insert({
                "ts": ts, "username": username, "action": action,
                "detail": detail, "tokens": rows, "cost_usd": cost,
            }).execute()
        else:
            con = _sqlite_conn()
            con.execute(
                "INSERT INTO usage_logs (ts,username,action,detail,tokens,cost_usd) VALUES (?,?,?,?,?,?)",
                (ts, username, action, detail, rows, cost),
            )
            con.commit(); con.close()
    except Exception:
        pass


def submit_feedback(username: str, page: str, rating: int, comment: str):
    ts = datetime.datetime.utcnow().isoformat()
    try:
        if _use_supabase():
            _supabase().table("feedback").insert({
                "ts": ts, "username": username, "page": page,
                "rating": rating, "comment": comment,
            }).execute()
        else:
            con = _sqlite_conn()
            con.execute(
                "INSERT INTO feedback (ts,username,page,rating,comment) VALUES (?,?,?,?,?)",
                (ts, username, page, rating, comment),
            )
            con.commit(); con.close()
    except Exception:
        pass


def get_usage_df() -> pd.DataFrame:
    try:
        if _use_supabase():
            res = _supabase().table("usage_logs").select("*").order("id", desc=True).limit(500).execute()
            return pd.DataFrame(res.data)
        else:
            con = _sqlite_conn()
            df  = pd.read_sql("SELECT * FROM usage_logs ORDER BY id DESC LIMIT 500", con)
            con.close(); return df
    except Exception:
        return pd.DataFrame()


def get_feedback_df() -> pd.DataFrame:
    try:
        if _use_supabase():
            res = _supabase().table("feedback").select("*").order("id", desc=True).limit(200).execute()
            return pd.DataFrame(res.data)
        else:
            con = _sqlite_conn()
            df  = pd.read_sql("SELECT * FROM feedback ORDER BY id DESC LIMIT 200", con)
            con.close(); return df
    except Exception:
        return pd.DataFrame()
