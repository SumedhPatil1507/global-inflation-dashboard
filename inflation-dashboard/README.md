# 🌍 Global Inflation Insights Dashboard

A production-grade, end-to-end economic analytics platform built with Streamlit, PyTorch, and the World Bank API. Analyzes post-COVID inflation dynamics across 20+ major economies with live data, machine learning models, anomaly detection, LSTM forecasting, and enterprise features like authentication, audit logs, cost tracking, and webhook notifications.

---

## Live Demo

🚀 **[Launch Dashboard](https://sumedhpatil1507-global-inflation-dashboard.streamlit.app)**

> Hosted free on [Streamlit Community Cloud](https://share.streamlit.io)

---

## Features

### Analytics
| Module | Description |
|---|---|
| EDA | Histograms, time-series trends, boxplots, violin plots, correlation heatmap, scatter matrix |
| ML Models | PyTorch feedforward neural network with loss curve + permutation feature importance + confidence score |
| Anomaly Detection | Z-score statistical detection + Autoencoder deep learning detection |
| Clustering | Hierarchical dendrogram + K-Means elbow method + labeled scatter |
| LSTM Forecasting | Per-country inflation forecast with configurable horizon + confidence band |
| Advanced Plots | 3D scatter, contour density, hexbin, facet grid |

### Enterprise
| Feature | Description |
|---|---|
| Authentication | SHA-256 hashed login, session management, logout |
| Data Editor | Live editable table with CSV + JSON export |
| Usage Logs | Every user action logged to SQLite with timestamp |
| Cost Tracking | Row-based cost model with daily cost chart |
| Feedback Loop | In-app rating + comment form, stored and reviewable |
| Webhooks | POST notifications to Slack, Discord, Make.com, or any HTTP endpoint |
| Admin Panel | Full audit dashboard — logs, feedback, cost — admin-only access |
| Citations | Inline data source attribution on every chart |
| Confidence Scores | Model and forecast confidence badges (green/amber/red) |

---

## Project Structure

```
inflation-dashboard/
├── app.py                        # Main Streamlit app — all tabs wired here
├── modules/
│   ├── data_loader.py            # World Bank API fetch + synthetic fallback
│   ├── eda.py                    # Interactive EDA (Plotly)
│   ├── models.py                 # Neural network training + feature importance
│   ├── anomaly.py                # Z-score + Autoencoder anomaly detection
│   ├── clustering.py             # Hierarchical + K-Means clustering
│   ├── forecasting.py            # LSTM per-country forecasting
│   ├── advanced_plots.py         # 3D, contour, hexbin, facet
│   ├── security.py               # Auth, session, login wall
│   ├── usage_logger.py           # SQLite logs, cost tracking, feedback, admin panel
│   └── webhooks.py               # Webhook dispatcher + sidebar settings UI
├── .streamlit/
│   └── secrets.toml              # Credentials + webhook URLs (never commit real secrets)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/SumedhPatil1507/global-inflation-dashboard.git
cd global-inflation-dashboard/inflation-dashboard

# 2. Install dependencies (use CPU torch to save ~1.5GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Default login credentials:
| Username | Password |
|---|---|
| `demo` | `demo123` |
| `admin` | `admin` |

---

## Data Sources

| Variable | Source | World Bank Indicator |
|---|---|---|
| Inflation Rate (CPI) | World Bank | `FP.CPI.TOTL.ZG` |
| GDP Growth | World Bank | `NY.GDP.MKTP.KD.ZG` |
| Unemployment Rate | World Bank | `SL.UEM.TOTL.ZS` |
| Lending Interest Rate | World Bank | `FR.INR.LEND` |
| Oil Price, Food Index, M2, Supply Chain | Synthetic proxy | — |

Data is fetched live via [wbgapi](https://pypi.org/project/wbgapi/). A fully synthetic fallback activates automatically if the API is unreachable.

---

## Configuration

### Credentials
Edit `.streamlit/secrets.toml`:
```toml
[users]
admin = "<sha256 of your password>"
demo  = "<sha256 of your password>"
```

Generate a hash:
```bash
python -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```

### Webhooks
Add your URLs to `.streamlit/secrets.toml`:
```toml
[webhooks]
forecast_done     = "https://hooks.slack.com/services/..."
anomaly_found     = "https://hooks.slack.com/services/..."
model_trained     = "https://hooks.slack.com/services/..."
feedback_received = "https://hooks.slack.com/services/..."
```

Supported destinations: Slack, Discord, Make.com, Zapier, webhook.site, or any HTTP POST endpoint.

Webhook payload format:
```json
{
  "event": "forecast_done",
  "timestamp": "2026-04-21T10:00:00",
  "user": "demo",
  "country": "USA",
  "horizon": 5
}
```

---

## ML Models

### Neural Network (Inflation Predictor)
- Architecture: 3-layer feedforward (64 → 32 → 1)
- Features: interest rate, oil price, GDP growth, unemployment, food price index, supply chain index
- Evaluation: MSE, R², confidence score badge
- Feature importance via permutation method

### LSTM (Forecasting)
- Architecture: 2-layer LSTM (hidden=64) with dropout
- Input: historical annual CPI series per country
- Output: N-year autoregressive forecast with ±1σ confidence band

### Autoencoder (Anomaly Detection)
- Architecture: encoder (6→32→8) + decoder (8→32→6)
- Anomalies flagged at configurable reconstruction error percentile

---

## Deployment

### Streamlit Community Cloud (free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → main file: `inflation-dashboard/app.py`
4. Add secrets in the Streamlit Cloud secrets UI (same format as `secrets.toml`)

### Update on GitHub
```bash
cd C:\Users\Sumedh\projects\global-inflation-dashboard
git add .
git commit -m "your message"
git push
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Visualization | Plotly, Matplotlib |
| ML / DL | PyTorch, scikit-learn |
| Data | World Bank API (wbgapi), pandas, numpy |
| Clustering | scipy |
| Auth | hashlib, hmac (SHA-256) |
| Storage | SQLite (usage.db) |
| Notifications | HTTP webhooks (requests) |

---

## Impact

- Covers 6 distinct ML/DL techniques in a single deployable app
- Enterprise patterns (auth, audit logs, cost tracking, webhooks) that most data science portfolios skip
- Live data integration — not a static CSV demo
- Real-world use cases: central bank monitoring, investment risk tracking, supply chain cost forecasting, policy research

---

## License

MIT — free to use, modify, and deploy.
