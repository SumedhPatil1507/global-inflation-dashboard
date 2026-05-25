# 🌍 Global Inflation Insights Dashboard

Production-grade economic analytics platform — live World Bank + FRED data, PyTorch ML, Supabase persistence, role-based auth, PDF export, and enterprise reporting.

## Live Demo

🚀 **[Launch Dashboard](https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app)**

> Hosted on [Streamlit Community Cloud](https://share.streamlit.io)

---

## What's Inside

### Analytics
| Module | Description |
|---|---|
| EDA | Sub-tabs: Data Diagnostics · Distributions · Correlations |
| Insights | Auto-generated Z-score flags, real rate gaps, deflationary signals |
| Trading Signals | Carry trade signals, regime switching allocator, country signal table |
| ML Models | PyTorch NN (sklearn Ridge fallback) · 80/20 split · permutation importance |
| Anomaly Detection | Z-score + Autoencoder (Z-score fallback when torch unavailable) |
| Clustering | Hierarchical dendrogram + K-Means elbow + scatter |
| LSTM Forecasting | Per-country forecast · confidence band · PDF export |
| Portfolio Stress Tester | Stagflation/hyperinflation/deflation/custom · Sharpe ratio · Monte Carlo · cumulative wealth · max drawdown |
| Advanced Plots | 3D scatter · contour density · hexbin · facet grid |

### Enterprise
| Feature | Description |
|---|---|
| Auth | SHA-256 login · rate limiting (5 attempts → 5min lockout) · role-based access |
| Roles | `admin` · `analyst` · `viewer` — each sees different tabs |
| Alert Thresholds | Configurable inflation/unemployment alerts → webhook fires on breach |
| Supabase DB | Persistent logs + feedback (SQLite fallback for local dev) |
| Usage Logs | Every action logged with timestamp, user, cost estimate |
| Cost Tracking | Daily cost chart in admin panel |
| Feedback Loop | In-app rating form, stored and reviewable by admin |
| Webhooks | POST to Slack, Discord, Make.com, Zapier on 5 event types |
| PDF Export | One-click country report with data table + charts + insights |
| Data Editor | Live editable table with CSV + JSON download |
| FRED API | Real monthly US data (CPI, Fed Funds Rate, Unemployment) |
| Docker | Single-command self-hosted deployment |
| CI/CD | GitHub Actions — lint + smoke test on every push |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│                   Streamlit (app.py)                            │
│  Auth · KPI Cards · 12 Tabs · Role-gated UI · PDF Export       │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────────┐
         ▼               ▼                   ▼
┌────────────────┐ ┌──────────────┐ ┌───────────────────┐
│  DATA LAYER    │ │  ML RUNTIME  │ │  ENTERPRISE LAYER │
│                │ │  (PyTorch /  │ │                   │
│ World Bank API │ │   sklearn)   │ │ Supabase Postgres │
│ FRED API       │ │              │ │ (usage_logs,      │
│ Synthetic      │ │ NN Predictor │ │  feedback)        │
│ fallback       │ │ LSTM Forecast│ │                   │
│                │ │ Autoencoder  │ │ SQLite (local dev)│
│ data_loader.py │ │ Stress Tester│ │ db.py             │
└────────────────┘ │ Monte Carlo  │ └───────────────────┘
                   └──────────────┘
         ┌───────────────┼───────────────────┐
         ▼               ▼                   ▼
┌────────────────┐ ┌──────────────┐ ┌───────────────────┐
│ SIGNAL LAYER   │ │ WEBHOOK      │ │  SECURITY LAYER   │
│                │ │ WORKER       │ │                   │
│ Regime Switch  │ │              │ │ SHA-256 + hmac    │
│ Carry Trade    │ │ Slack        │ │ Rate limiting     │
│ Yield Optimizer│ │ Discord      │ │ Role-based access │
│ Signal Table   │ │ Make.com     │ │ Session state     │
│                │ │ Zapier       │ │                   │
│trading_signals │ │ webhooks.py  │ │ security.py       │
└────────────────┘ └──────────────┘ └───────────────────┘
```

---

## Project Structure

```
inflation-dashboard/
├── app.py                     # Main app — all tabs, auth, KPIs
├── modules/
│   ├── data_loader.py         # World Bank + FRED API + synthetic fallback
│   ├── db.py                  # Supabase (prod) / SQLite (dev) persistence layer
│   ├── security.py            # Auth, rate limiting, roles, login wall
│   ├── insights.py            # Business callouts, alert thresholds, PDF export
│   ├── usage_logger.py        # Admin panel (delegates to db.py)
│   ├── webhooks.py            # Webhook dispatcher + sidebar UI
│   ├── models.py              # NN training (train/test split, persistence)
│   ├── anomaly.py             # Z-score + Autoencoder
│   ├── clustering.py          # Hierarchical + K-Means
│   ├── forecasting.py         # LSTM per-country forecast
│   ├── eda.py                 # Interactive EDA (Plotly)
│   └── advanced_plots.py      # 3D, contour, hexbin, facet
├── .streamlit/
│   ├── config.toml            # Theme (dark, brand colors)
│   └── secrets.toml           # Credentials + API keys (never commit real values)
├── Dockerfile                 # Self-hosted deployment
├── requirements.txt           # Pinned dependencies
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/SumedhPatil1507/global-inflation-dashboard.git
cd global-inflation-dashboard/inflation-dashboard

# CPU-only torch (saves ~1.5GB)
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run app.py
```

Default credentials:
| Username | Password | Role |
|---|---|---|
| `demo` | `demo123` | analyst |
| `admin` | `admin` | admin |

---

## Docker

```bash
cd inflation-dashboard
docker build -t inflation-dashboard .
docker run -p 8501:8501 inflation-dashboard
# Open http://localhost:8501
```

---

## Configuration

### Add a new user
```bash
python -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```
Add to `.streamlit/secrets.toml`:
```toml
[users]
yourname = "thehash:analyst"   # or :admin or :viewer
```

### Enable Supabase (persistent DB)
1. Create project at [supabase.com](https://supabase.com)
2. Run in Supabase SQL editor:
```sql
create table usage_logs (
  id serial primary key, ts text, username text,
  action text, detail text, tokens int default 0, cost_usd float default 0
);
create table feedback (
  id serial primary key, ts text, username text,
  page text, rating int, comment text
);
```
3. Add to `secrets.toml`:
```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key"
```

### Enable FRED API (real US monthly data)
1. Get free key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Add to `secrets.toml`:
```toml
[fred]
api_key = "your_fred_api_key"
```

### Enable Webhooks
```toml
[webhooks]
forecast_done     = "https://hooks.slack.com/services/..."
anomaly_found     = "https://hooks.slack.com/services/..."
model_trained     = "https://hooks.slack.com/services/..."
feedback_received = "https://hooks.slack.com/services/..."
threshold_breach  = "https://hooks.slack.com/services/..."
```

---

## Data Sources

| Variable | Source | Indicator |
|---|---|---|
| Inflation (CPI) | World Bank | `FP.CPI.TOTL.ZG` |
| GDP Growth | World Bank | `NY.GDP.MKTP.KD.ZG` |
| Unemployment | World Bank | `SL.UEM.TOTL.ZS` |
| Lending Rate | World Bank | `FR.INR.LEND` |
| US CPI (monthly) | FRED | `CPIAUCSL` |
| US Fed Funds Rate | FRED | `FEDFUNDS` |
| US Unemployment | FRED | `UNRATE` |
| Oil, Food, M2, Supply Chain | Synthetic proxy | — |

---

## ML Models

| Model | Architecture | Evaluation |
|---|---|---|
| Inflation Predictor | 3-layer NN (64→32→1) + Dropout | 80/20 train/test split, MSE + R² |
| LSTM Forecaster | 2-layer LSTM (hidden=64) + Dropout | Autoregressive, ±1σ confidence band |
| Anomaly Autoencoder | Encoder (6→32→8) + Decoder | Reconstruction error percentile threshold |

---

## Deployment — Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → Create app
3. Repo: `SumedhPatil1507/global-inflation-dashboard` · File: `inflation-dashboard/app.py`
4. Advanced settings → paste `secrets.toml` contents → Deploy

### Update on GitHub
```bash
cd C:\Users\Sumedh\projects\global-inflation-dashboard
git add .
git commit -m "your message"
git push
```

---

## Tech Stack

Streamlit · PyTorch · Plotly · Supabase · FRED API · World Bank API · scikit-learn · scipy · ReportLab · Docker · GitHub Actions

---

## License

MIT
