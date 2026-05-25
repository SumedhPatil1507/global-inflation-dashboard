"""
Authentication — two modes:
  1. JWT via FastAPI backend (when BACKEND_URL is set in secrets.toml)
  2. Local SHA-256 fallback (Streamlit Cloud / no backend)

Rate limiting, role-based access, and session management in both modes.
"""
import hashlib
import hmac
import time
import streamlit as st

ROLES = {
    "admin":   ["eda","models","anomaly","clustering","forecasting",
                "advanced","editor","feedback","admin"],
    "analyst": ["eda","models","anomaly","clustering","forecasting",
                "advanced","editor","feedback"],
    "viewer":  ["eda","clustering","advanced"],
}

_DEFAULT_USERS = {
    "admin": ("8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918", "admin"),
    "demo":  ("d3ad9315b7be5dd53b31a273b3b3aba5defe700808305aa16a3062b76658a791", "analyst"),
}

MAX_ATTEMPTS = 5
LOCKOUT_SECS = 300


def _backend_url() -> str:
    try:
        return st.secrets.get("backend", {}).get("url", "")
    except Exception:
        return ""


def _jwt_login(username: str, password: str) -> dict | None:
    """Authenticate via FastAPI JWT endpoint."""
    import requests
    url = _backend_url()
    if not url:
        return None
    try:
        r = requests.post(
            f"{url}/api/auth/token",
            data={"username": username, "password": password},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            return {
                "token":    data["access_token"],
                "role":     data["role"],
                "username": data["username"],
            }
    except Exception:
        pass
    return None


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _get_local_users() -> dict:
    try:
        raw = dict(st.secrets.get("users", {}))
        if raw:
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
    attempts = st.session_state.get("login_attempts", {})
    entry    = attempts.get(username, {"count": 0, "since": 0})
    if entry["count"] >= MAX_ATTEMPTS:
        remaining = int(LOCKOUT_SECS - (time.time() - entry["since"]))
        if remaining > 0:
            return True, remaining
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
    if st.session_state.get("authenticated"):
        return

    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem">
        <h1 style="color:#38bdf8;font-size:2.4rem">🌍 Global Inflation Insights</h1>
        <p style="color:#94a3b8;font-size:1rem;max-width:580px;margin:.5rem auto 0">
            Production-grade economic analytics — live FRED + World Bank data,
            ML forecasting, portfolio stress testing, and enterprise reporting.
        </p>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        mode = "JWT (FastAPI)" if _backend_url() else "Local"
        st.caption(f"Auth mode: **{mode}**")
        with st.form("login_form", clear_on_submit=False):
            username  = st.text_input("Username", placeholder="demo")
            password  = st.text_input("Password", type="password", placeholder="demo123")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            locked, secs = _check_lockout(username)
            if locked:
                st.error(f"Account locked. Try again in {secs}s.")
            else:
                authenticated = False

                # ── Try JWT backend first ─────────────────────────────────
                if _backend_url():
                    result = _jwt_login(username, password)
                    if result:
                        _record_attempt(username, True)
                        st.session_state.update({
                            "authenticated": True,
                            "username":      result["username"],
                            "role":          result["role"],
                            "jwt_token":     result["token"],
                            "auth_mode":     "jwt",
                        })
                        authenticated = True
                        st.rerun()

                # ── Local SHA-256 fallback ────────────────────────────────
                if not authenticated:
                    users  = _get_local_users()
                    hashed = _hash(password)
                    entry  = users.get(username)
                    if entry and hmac.compare_digest(entry[0], hashed):
                        _record_attempt(username, True)
                        st.session_state.update({
                            "authenticated": True,
                            "username":      username,
                            "role":          entry[1],
                            "auth_mode":     "local",
                        })
                        st.rerun()
                    else:
                        _record_attempt(username, False)
                        left = MAX_ATTEMPTS - st.session_state.get(
                            "login_attempts", {}).get(username, {}).get("count", 0)
                        st.error(f"Invalid credentials. {max(0,left)} attempt(s) remaining.")

        st.caption("Demo: `demo` / `demo123`  ·  Admin: `admin` / `admin`")
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
    return tab in ROLES.get(current_role(), [])


def get_jwt_token() -> str:
    """Return JWT token for backend API calls, or empty string."""
    return st.session_state.get("jwt_token", "")


def api_headers() -> dict:
    """Auth headers for FastAPI calls."""
    token = get_jwt_token()
    return {"Authorization": f"Bearer {token}"} if token else {}
