"""
Authentication, data security, and session management.
Uses st.secrets for credentials — set in .streamlit/secrets.toml
"""
import hashlib
import hmac
import streamlit as st

_DEFAULT_USERS = {
    "admin": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin
    "demo":  "2a97516c354b68848cdbd8f54a226a0a55b21ed138e207ad6c5cbb9c00aa5aea",  # demo123
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _get_users() -> dict:
    try:
        return dict(st.secrets.get("users", _DEFAULT_USERS))
    except Exception:
        return _DEFAULT_USERS


def login_wall():
    """Renders login form and blocks app if not authenticated."""
    if st.session_state.get("authenticated"):
        return  # already logged in — let app continue

    st.markdown("## 🔐 Login to Global Inflation Dashboard")
    st.info("Demo credentials — username: `demo`  password: `demo123`")

    with st.form("login_form"):
        username  = st.text_input("Username")
        password  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    # Only validate after the user actually clicks submit
    if submitted:
        users  = _get_users()
        hashed = _hash(password)
        if username in users and hmac.compare_digest(users[username], hashed):
            st.session_state["authenticated"] = True
            st.session_state["username"]      = username
            st.rerun()
        else:
            st.error("Invalid credentials. Try username: demo / password: demo123")

    # Block the rest of the app until authenticated
    st.stop()


def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()


def current_user() -> str:
    return st.session_state.get("username", "anonymous")
