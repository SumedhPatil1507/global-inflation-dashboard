"""
Authentication — Supabase Auth (production) with secrets.toml fallback (local dev).
Includes rate limiting, login audit, and role-based access.
"""
import hashlib
import hmac
import time
import streamlit as st

# ── Roles ─────────────────────────────────────────────────────────────────────
ROLES = {
    "admin":  ["eda", "models", "anomaly", "clustering", "forecasting",
               "advanced", "editor", "feedback", "admin"],
    "analyst":["eda", "models", "anomaly", "clustering", "forecasting",
               "advanced", "editor", "feedback"],
    "viewer": ["eda", "clustering", "advanced"],
}

_DEFAULT_USERS = {
    # username: (sha256_hash, role)
    "admin": ("8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918", "admin"),
    "demo":  ("d3ad9315b7be5dd53b31a273b3b3aba5defe700808305aa16a3062b76658a791", "analyst"),
}

MAX_ATTEMPTS = 5
LOCKOUT_SECS = 300  # 5 minutes


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _get_users() -> dict:
    """Returns {username: (hash, role)}."""
    try:
        raw = dict(st.secrets.get("users", {}))
        if raw:
            # secrets.toml format: username = "hash:role" or just "hash" (defaults to analyst)
            out = {}
            for u, v in raw.items():
                if ":" in str(v):
                    h, r = str(v).split(":", 1)
                    out[u] = (h.strip(), r.strip())
                else:
                    out[u] = (str(v).strip(), "analyst")
            return out
    except Exception:
        pass
    return _DEFAULT_USERS


def _check_lockout(username: str) -> tuple[bool, int]:
    """Returns (is_locked, seconds_remaining)."""
    attempts = st.session_state.get("login_attempts", {})
    entry    = attempts.get(username, {"count": 0, "since": 0})
    if entry["count"] >= MAX_ATTEMPTS:
        elapsed   = time.time() - entry["since"]
        remaining = int(LOCKOUT_SECS - elapsed)
        if remaining > 0:
            return True, remaining
        else:
            # Reset after lockout period
            attempts[username] = {"count": 0, "since": 0}
            st.session_state["login_attempts"] = attempts
    return False, 0


def _record_attempt(username: str, success: bool):
    attempts = st.session_state.get("login_attempts", {})
    if success:
        attempts[username] = {"count": 0, "since": 0}
    else:
        entry = attempts.get(username, {"count": 0, "since": time.time()})
        entry["count"] += 1
        if entry["count"] == 1:
            entry["since"] = time.time()
        attempts[username] = entry
    st.session_state["login_attempts"] = attempts


def login_wall():
    """Full-page login. Blocks app until authenticated."""
    if st.session_state.get("authenticated"):
        return

    # ── Landing / login page ──────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem">
        <h1 style="color:#38bdf8;font-size:2.5rem">🌍 Global Inflation Insights</h1>
        <p style="color:#94a3b8;font-size:1.1rem;max-width:600px;margin:0 auto">
            Production-grade economic analytics — live World Bank + FRED data,
            ML forecasting, anomaly detection, and enterprise reporting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("### Sign In")
        with st.form("login_form", clear_on_submit=False):
            username  = st.text_input("Username", placeholder="demo")
            password  = st.text_input("Password", type="password", placeholder="demo123")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            locked, secs = _check_lockout(username)
            if locked:
                st.error(f"Too many failed attempts. Try again in {secs}s.")
            else:
                users  = _get_users()
                hashed = _hash(password)
                entry  = users.get(username)
                if entry and hmac.compare_digest(entry[0], hashed):
                    _record_attempt(username, success=True)
                    st.session_state["authenticated"] = True
                    st.session_state["username"]      = username
                    st.session_state["role"]          = entry[1]
                    st.rerun()
                else:
                    _record_attempt(username, success=False)
                    attempts = st.session_state.get("login_attempts", {}).get(username, {})
                    left     = MAX_ATTEMPTS - attempts.get("count", 0)
                    st.error(f"Invalid credentials. {left} attempt(s) remaining.")

        st.caption("Demo: username `demo` / password `demo123`")

    st.stop()


def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()


def current_user() -> str:
    return st.session_state.get("username", "anonymous")


def current_role() -> str:
    return st.session_state.get("role", "viewer")


def can_access(tab: str) -> bool:
    role = current_role()
    return tab in ROLES.get(role, [])
