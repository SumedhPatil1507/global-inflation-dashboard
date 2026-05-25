# 🌍 Global Inflation Insights Dashboard

Production-grade economic analytics platform with a **FastAPI async backend**, **Celery + Redis task queue**, **JWT stateless auth**, **real live data** (FRED + yfinance + World Bank), and a **Streamlit frontend** with 12 analytical tabs.

## Live Demo

🚀 **[Launch Dashboard](https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app)**

> Hosted on [Streamlit Community Cloud](https://share.streamlit.io)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                              │
│                    Streamlit  (app.py)                               │
│   JWT Auth · 12 Tabs · Role-gated UI · PDF Export · Webhooks        │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  HTTP (httpx / requests)
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND  (:8000)                        │
│  /api/auth/token  →  JWT issue                                       │
│  /api/data/combined  →  FRED + yfinance + World Bank merge           │
│  /api/ml/train       →  enqueue Celery task → return task_id         │
│  /api/ml/forecast    →  enqueue Celery task → return task_id         │
│  /api/ml/monte-carlo →  enqueue Celery task → return task_id         │
│  /api/ml/tasks/{id}  →  poll result                                  │
└──────────┬────────────────────────────────────────────────┬──────────┘
           │  Celery tasks                                  │  SQL
           ▼                                                ▼
┌─────────────────────────┐                   ┌────────────────────────┐
│   CELERY WORKER         │                   │  SUPABASE POSTGRES     │
│   (async ML runtime)    │                   │  usage_logs            │
│                         │                   │  feedback              │
│  PyTorch NN training    │                   │  (SQLite fallback      │
│  LSTM forecasting       │                   │   for local dev)       │
│  Monte Carlo sim        │                   └────────────────────────┘
│  Autoencoder anomaly    │
└──────────┬──────────────┘
           │  broker + results
           ▼
┌─────────────────────────┐
│   REDIS  (:6379)        │
│   Task queue            │
│   Result backend        │
└─────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│                                                                      │
│  FRED API          → CPI, Fed Funds Rate, Unemployment, M2 (real)   │
│  yfinance          → WTI Oil, Gold futures, DBA food ETF (real)     │
│  World Bank API    → 19 countries, 4 macro indicators (annual)      │
│  Synthetic fallback→ when all APIs unavailable                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      SECURITY LAYER                                  │
│                                                                      │
│  JWT (python-jose)  → stateless tokens, 60min expiry, refresh       │
│  bcrypt             → password hashing in FastAPI backend            │
│  SHA-256 + hmac     → local fallback (Streamlit Cloud, no backend)  │
│  Rate limiting      → 5 attempts → 5min lockout (session state)     │
│  Role-based access  → admin · analyst · viewer                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Analytics (12 Tabs)
| Tab | Description |
|---|---|
| EDA | Sub-tabs: Data Diagnostics · Distributions · Correlations |
| Insights | Auto Z-score flags, real rate gaps, deflationary signals |
| Trading Signals | Carry trade signals, regime switching allocator, country signal table |
| ML Models | PyTorch NN / sklearn Ridge · 80/20 split · permutation importance |
| Anomaly Detection | Z-score + Autoencoder (graceful fallback without torch) |
| Clustering | Hierarchical dendrogram + K-Means elbow + scatter |
| LSTM Forecasting | Per-country forecast · confidence band · PDF export |
| Portfolio Stress Tester | Stagflation/hyperinflation/deflation · Sharpe · Monte Carlo · drawdown |
| Advanced Plots | 3D scatter · contour density · hexbin · facet grid |
| Data Editor | Live editable table · CSV + JSON export |
| Feedback | In-app rating form |
| Admin | Usage logs · cost tracking · feedback review |

### Enterprise
| Feature | Description |
|---|---|
| JWT Auth | FastAPI issues tokens · Streamlit validates · 60min expiry + refresh |
| Local fallback | SHA-256 + hmac when no backend configured |
| Rate limiting | 5 failed attempts → 5min lockout |
| Role-based access | admin · analyst · viewer — tab-level gating |
| Async ML | Celery + Redis queues heavy LSTM/Monte Carlo jobs |
| Real data | FRED (M2, CPI, rates) + yfinance (Oil, Gold, Food) — no synthetic proxies |
| Supabase DB | Persistent logs + feedback (SQLite fallback) |
| Webhooks | Slack/Discord/Make.com on 5 event types |
| PDF Export | Country report with charts + insights |
| Docker | Full stack via docker-compose |
| CI/CD | GitHub Actions — frontend + backend lint + smoke tests |

---

## Project Structure

```
.
├── backend/                        # FastAPI async backend
│   ├── main.py                     # App entry point
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── core/
│   │   ├── config.py               # Pydantic settings (env vars)
│   │   └── security.py             # JWT issue/verify, bcrypt, role guards
│   ├── api/routers/
│   │   ├── auth.py                 # POST /api/auth/token, GET /api/auth/me
│   │   ├── data.py                 # GET /api/data/combined, /countries
│   │   └── ml.py                   # POST /api/ml/train|forecast|monte-carlo
│   ├── services/
│   │   ├── data_service.py         # FRED + yfinance + World Bank fetchers
│   │   └── ml_service.py           # PyTorch NN, LSTM, Monte Carlo
│   └── workers/
│       ├── celery_app.py           # Celery + Redis config
│       └── tasks.py                # Async task definitions
│
├── inflation-dashboard/            # Streamlit frontend
│   ├── app.py                      # Main app — 12 tabs
│   ├── modules/
│   │   ├── security.py             # JWT + local auth, rate limiting, roles
│   │   ├── data_loader.py          # FRED + yfinance + WB + synthetic fallback
│   │   ├── db.py                   # Supabase / SQLite persistence
│   │   ├── insights.py             # Business callouts, alerts, PDF export
│   │   ├── trading_signals.py      # Regime switching, carry trade, yield optimizer
│   │   ├── forecasting.py          # LSTM + stress tester + Monte Carlo plots
│   │   ├── models.py               # NN training, feature importance
│   │   ├── anomaly.py              # Z-score + Autoencoder
│   │   ├── clustering.py           # Hierarchical + K-Means
│   │   ├── eda.py                  # EDA plots
│   │   ├── advanced_plots.py       # 3D, contour, hexbin, facet
│   │   ├── usage_logger.py         # Admin panel
│   │   └── webhooks.py             # Webhook dispatcher
│   ├── .streamlit/
│   │   ├── config.toml             # Dark theme
│   │   └── secrets.toml            # Credentials + API keys
│   ├── Dockerfile
│   ├── requirements.txt
│   └── runtime.txt                 # python-3.11.9
│
├── docker-compose.yml              # Full stack: api + worker + redis + frontend
├── .env.example                    # Environment variable template
├── .github/workflows/ci.yml        # CI: frontend + backend lint + smoke tests
└── README.md
```

---

## Quickstart — Local (Streamlit only)

```bash
git clone https://github.com/SumedhPatil1507/global-inflation-dashboard.git
cd global-inflation-dashboard/inflation-dashboard

pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run app.py
```

Credentials: `demo` / `demo123` · `admin` / `admin`

---

## Quickstart — Full Stack (Docker)

```bash
cp .env.example .env
# Edit .env — add FRED_API_KEY, JWT_SECRET_KEY, SUPABASE_URL/KEY

docker-compose up --build
# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
```

---

## Configuration

### secrets.toml (Streamlit Cloud)
```toml
[users]
admin = "sha256hash:admin"
demo  = "sha256hash:analyst"

[backend]
url = "https://your-api.railway.app"   # enables JWT auth

[fred]
api_key = "your_fred_api_key"          # enables real M2, CPI, rates

[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key"

[webhooks]
forecast_done = "https://hooks.slack.com/services/..."
```

### Supabase tables
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

---

## Data Sources

| Variable | Source | Real / Synthetic |
|---|---|---|
| CPI Inflation | FRED `CPIAUCSL` | ✅ Real |
| Fed Funds Rate | FRED `FEDFUNDS` | ✅ Real |
| Unemployment | FRED `UNRATE` | ✅ Real |
| GDP Growth | FRED `A191RL1Q225SBEA` | ✅ Real |
| Money Supply M2 | FRED `M2SL` | ✅ Real |
| WTI Oil Price | yfinance `CL=F` | ✅ Real |
| Gold Price | yfinance `GC=F` | ✅ Real |
| Food Price Index | yfinance `DBA` | ✅ Real (ETF proxy) |
| 19-country macro | World Bank API | ✅ Real |
| Supply Chain Index | Hardcoded 2020-2024 estimates | ⚠️ Approximate |

---

## Update on GitHub

```bash
cd C:\Users\Sumedh\projects\global-inflation-dashboard
git add .
git commit -m "feat: FastAPI backend, Celery+Redis, JWT auth, real FRED+yfinance data"
git push
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Task Queue | Celery + Redis |
| ML/DL | PyTorch · scikit-learn |
| Auth | JWT (python-jose) · bcrypt · SHA-256 fallback |
| Data | FRED API · yfinance · World Bank (wbgapi) |
| Database | Supabase Postgres · SQLite (dev) |
| Visualization | Plotly · Matplotlib |
| Infrastructure | Docker · docker-compose · GitHub Actions |

---

## License

MIT
