"""
Webhook dispatcher — sends POST payloads to configured URLs on key events.
Configure webhook URLs in .streamlit/secrets.toml:
  [webhooks]
  forecast_done = "https://hooks.slack.com/services/..."
  anomaly_found = "https://your-endpoint.com/webhook"
"""
import requests
import datetime
import streamlit as st


def _get_urls() -> dict:
    try:
        return dict(st.secrets.get("webhooks", {}))
    except Exception:
        return {}


def fire(event: str, payload: dict, timeout: int = 5):
    """
    Fire a webhook for a named event.
    Silently swallows errors so the app never crashes on webhook failure.
    """
    urls = _get_urls()
    url = urls.get(event)
    if not url:
        return
    body = {
        "event": event,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        **payload,
    }
    try:
        requests.post(url, json=body, timeout=timeout)
    except Exception:
        pass


def render_webhook_settings():
    """UI to test webhook endpoints from the sidebar."""
    st.markdown("### 🔗 Webhook Settings")
    st.caption("Configure URLs in `.streamlit/secrets.toml` under `[webhooks]`.")
    urls = _get_urls()
    if urls:
        for event, url in urls.items():
            masked = url[:30] + "…" if len(url) > 30 else url
            st.markdown(f"- **{event}**: `{masked}`")
        if st.button("🧪 Test Webhooks"):
            for event in urls:
                fire(event, {"test": True, "source": "dashboard"})
            st.success("Test payloads sent.")
    else:
        st.info("No webhooks configured yet.")
