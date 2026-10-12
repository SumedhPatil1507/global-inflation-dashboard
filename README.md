# 🌐 Global Inflation Insights & Macro Intelligence Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Step%20Agent-blue.svg)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Security](https://img.shields.io/badge/Auth-RS256%20JWT%20%2B%20RBAC-green.svg)](https://jwt.io/)

A production-grade macroeconomic intelligence platform combining an asynchronous **FastAPI** backend, **LangGraph** autonomous research agent, **Dense RAG** knowledge retrieval over central bank publications, **Quantitative Portfolio Backtesting**, and a rich **Streamlit** multi-tab visualization dashboard with **fully interactive Plotly charts** and **Streamlit-ready backend integration**.

---

## 🚀 Live Cloud Deployment

Access the interactive live Streamlit dashboard directly in your browser:
👉 **[https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app/](https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app/)**

---

## 🏛️ System Architecture

```
                                  ┌──────────────────────────────────────────────┐
                                  │           Streamlit Frontend (UI)           │
                                  │  - 13 Specialized Analytical Tabs           │
                                  │  - Fully Interactive Plotly Visualizations │
                                  │  - LangGraph Macro Agent Interactive Console │
                                  │  - Streamlit-Ready Backend Integration     │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         │ HTTP REST / RS256 JWT
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │            FastAPI Backend Engine            │
                                  │  - RS256 Scoped JWT Auth & RBAC Middleware   │
                                  │  - Immutable Audit Trail (Ledger)            │
                                  │  - CORS & Rate-Limiting Controls             │
                                  │  - Streamlit-Ready Import Handling           │
                                  └──────┬───────────────┬───────────────┬───────┘
                                         │               │               │
                 ┌───────────────────────┘               │               └───────────────────────┐
                 ▼                                       ▼                                       ▼
  ┌──────────────────────────────┐        ┌──────────────────────────────┐        ┌──────────────────────────────┐
  │   LangGraph Macro Agent      │        │    Quantitative Engine       │        │     Data & Model Services    │
  │  1. Planner & Decomposition  │        │  - Multi-Asset Backtest      │        │  - FRED + World Bank APIs    │
  │  2. Macro Time-Series API    │        │  - Sharpe / Max Drawdown     │        │  - PyTorch NN Inflation Reg  │
  │  3. RAG Central Bank Corpus  │        │  - Equity & Underwater Curves│        │  - Celery Worker / Redis Q   │
  │  4. Research Memo Synthesis  │        │  - Dynamic Asset Allocation  │        │  - TimescaleDB / PostgreSQL  │
  │  5. World Bank/RBI Reports  │        │  - Streamlit-Ready Backtest  │        │  - Graceful API Fallbacks    │
  └──────────────────────────────┘        └──────────────────────────────┘        └──────────────────────────────┘
```

---

## ✨ Key Features & Analytical Modules

| Tab # | Module | Core Capabilities |
|---|---|---|
| **Tab 0** | **📊 Exploratory Data Analysis (EDA)** | Boxplots, violin distributions, metric trendlines, correlation matrices, and regional heatmaps across 18+ global economies. |
| **Tab 1** | **💡 Automated Business Insights** | Auto-synthesized executive takeaways, real interest rate spreads ($r = i - \pi$), and historical percentile rankings. |
| **Tab 2** | **🔬 Macro Research Agent (LangGraph + RAG)** | Autonomous multi-step economic reasoning engine with tool access to FRED/World Bank data endpoints, quantitative portfolio backtesters, and dense RAG over World Bank & RBI publications. |
| **Tab 3** | **📈 Trading Signals & Regimes** | Inflation-adjusted yield carry trade optimizer and 4-quadrant regime switching allocator (Stagflation, Goldilocks, Reflation, Deflation). |
| **Tab 4** | **🤖 Machine Learning Models** | Deep PyTorch Neural Network / Ridge regression with held-out test evaluation, loss curves, and permutation feature importance. |
| **Tab 5** | **🔍 Anomaly Detection** | Isolation Forest multivariate anomaly scoring highlighting geopolitical and pandemic inflation spikes. |
| **Tab 6** | **🔬 Macro Clustering** | Unsupervised K-Means clustering identifying macroeconomic archetypes and policy regimes. |
| **Tab 7** | **🔮 Time-Series Forecasting** | Statistical ARIMA, Facebook Prophet, and Vector Autoregression (VAR) multi-country co-movement modeling. |
| **Tab 8** | **💥 Macro Stress Testing** | Scenario simulation engine (Energy Shock, Supply Disruption, Monetary Tightening, Stagflation). |
| **Tab 9** | **🎨 Advanced Visualizations** | Radar charts, 3D metric scatter plots, and country-by-country facet grids. |
| **Tab 10** | **✏️ Interactive Data Editor** | In-browser dataframe editing with CSV and JSON data export capabilities. |
| **Tab 11** | **💬 User Feedback & Audit** | In-app feedback loop with real-time state telemetry. |
| **Tab 12** | **🛡️ Admin Panel & Audit Ledger** | User role administration and cryptographically signed audit log inspector. |

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit, Plotly, Seaborn, Matplotlib, ReportLab (PDF Export)
- **Backend:** FastAPI, Uvicorn, Pydantic v2, Python-Jose (RS256), Cryptography
- **Orchestration & Agents:** LangGraph, StateGraph, Custom Semantic Vector RAG
- **Quantitative & ML:** PyTorch, Scikit-Learn, NumPy, Pandas, SciPy, Statsmodels
- **Data Integrations:** Federal Reserve Economic Data (FRED), World Bank API (`wbgapi`), Yahoo Finance (`yfinance`)
- **Infrastructure & Storage:** PostgreSQL / TimescaleDB, Redis, Celery, Docker & Docker Compose

---

## 🚦 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/SumedhPatil1507/global-inflation-dashboard.git
cd global-inflation-dashboard
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies for dashboard and backend
pip install -r inflation-dashboard/requirements.txt
pip install -r backend/requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run inflation-dashboard/app.py
```
The application will launch at `http://localhost:8501`.

The repository-root `requirements.txt` includes the dashboard dependency set and is used by Streamlit Community Cloud. To deploy, connect this GitHub repository in Community Cloud, select `main` as the branch, and set the app file to `inflation-dashboard/app.py`. Add any required secrets (for example, live data or Supabase credentials) in the app's **Settings → Secrets** panel. The dashboard remains usable in synthetic demo mode without external data credentials.

### 4. (Optional) Run the FastAPI Backend Server
```bash
uvicorn backend.main:app --reload --port 8000
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

### 5. (Optional) Full Docker Compose Deployment
```bash
docker-compose up --build -d
```

---

## 🔐 Authentication, RBAC & Scopes

The API and Dashboard employ asymmetric **RS256 JWT** authentication with granular scope-based permissions:

| Role | JWT Scopes Assigned | Accessible Dashboard Tabs |
|---|---|---|
| **Admin** | `admin:all`, `macro:research`, `macro:read`, `quant:backtest`, `ml:train` | All Tabs (0 – 12) + Admin Panel |
| **Analyst** | `macro:research`, `macro:read`, `quant:backtest`, `ml:train` | Tabs 0 – 11 |
| **Viewer** | `macro:read` | EDA, Insights, Macro Agent, Clustering, Advanced |

*Note: Demo logins are configured out-of-the-box (`admin` / `analyst` / `viewer`). RSA keypairs (`certs/private_key.pem` and `certs/public_key.pem`) are automatically provisioned on launch if absent.*

---

## 📡 Key REST API Endpoints

| Method | Endpoint | Scopes / Roles | Description |
|---|---|---|---|
| `POST` | `/api/auth/token` | Public | Obtain RS256 JWT access token |
| `POST` | `/api/macro/research` | `macro:research` / Analyst | Execute LangGraph Macro Research Agent with multi-step reasoning |
| `GET` | `/api/macro/rag/reports` | `macro:research` / Analyst | Semantic search over World Bank and RBI research corpus |
| `POST` | `/api/quant/backtest` | `quant:backtest` / Analyst | Multi-asset quantitative inflation-hedge portfolio backtest |
| `POST` | `/api/ml/train` | `ml:train` / Analyst | Asynchronous model training task via Celery |
| `POST` | `/api/ml/predict` | `macro:read` / Viewer | Neural network inflation forecast inference |
| `GET` | `/health` | Public | System health check & active feature flags |

## Macro Agent Guardrails, Tracing & Evaluation

Research memorandums now pass through a local Guardrails AI validator before they are returned. The validator checks that cited and statutory/section references occur in retrieved source material, and blocks the entire memorandum if a forward-looking policy-rate claim is not directly supported by retrieved evidence. Failed validation returns a safe withheld notice with issue details in the API `validation` field; raw invalid text is not delivered.

The agent emits OpenTelemetry spans for the top-level run, each LangGraph state transition, RAG retrieval, quantitative backtest execution, and memorandum validation. Install from `backend/requirements.txt` (also listed in the dashboard requirements for its in-process agent UI). Configure an OTLP collector with standard OpenTelemetry variables, for example:

```powershell
$env:OTEL_SERVICE_NAME = "global-inflation-dashboard"
$env:OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = "http://localhost:4318/v1/traces"
uvicorn backend.main:app --reload --port 8000
```

When no endpoint is configured, spans are still created by the SDK but are not exported. Set `OTEL_TRACES_EXPORTER=console` for local span inspection.

The offline audit runner evaluates **50 fixed macro scenarios** against the production retriever and memorandum synthesizer and computes RAGAS Faithfulness, Context Precision, and Answer Relevancy. RAGAS requires a judge LLM and embeddings API key; install its separate dependencies and provide `OPENAI_API_KEY`:

```powershell
pip install -r eval/requirements.txt
$env:OPENAI_API_KEY = "..."
python eval/ragas_macro_eval.py
```

The script writes a timestamped JSON audit under `eval/reports/`. Select models with `RAGAS_LLM_MODEL` and `RAGAS_EMBEDDING_MODEL`, control request concurrency with `RAGAS_MAX_CONCURRENCY`, or set `--output path/to/report.json`.

---

## 🆕 Recent Updates & Enhancements

### ✅ Macro Research Agent (LangGraph + RAG)
- **Fully implemented** LangGraph-based autonomous multi-step reasoning engine
- **RAG Knowledge Base** with chunked and embedded World Bank/RBI reports
- **JWT-scoped access** with `macro:research` scope requirement
- **Interactive multi-tab interface** with research memorandum synthesis
- **Quantitative backtesting integration** for inflation-hedge portfolios

### ✅ Streamlit-Ready Backend Integration
- **Graceful import handling** for FastAPI dependencies in Streamlit environment
- **Fallback mechanisms** when external APIs are unavailable
- **Zero-dependency runtime** for macro agent execution
- **Compatible with both** standalone Streamlit and full FastAPI deployment

### ✅ Fully Interactive Visualizations
- **All plots converted to Plotly** for full interactivity
- **3D scatter plots** with zoom, pan, and rotation
- **Interactive contour density plots** replacing static hexbin
- **Real-time hover tooltips** and data exploration
- **Responsive cluster visualizations** with interactive dendrogram alternatives

### ✅ Robust Anomaly Detection & Model Fallbacks
- **Autoencoder & Z-Score anomaly detectors** with dynamic percentile and contamination thresholding
- **Zero-failure fallback** on CPU and environments without PyTorch (automatic Z-score reconstruction error proxy)
- **Defensive column handling** and Plotly chart rendering for standalone and cloud deployments

### ✅ Enhanced Security & Authentication
- **RS256 JWT-scoped access** for macro research endpoints
- **Role-based access control** (Admin, Analyst, Viewer)
- **Automatic RSA key generation** for development environments
- **Streamlit-compatible security** functions for frontend integration

---

## Mixed-Frequency Nowcasting & Execution-Aware Backtesting

The quantitative module includes a release-time-safe MIDAS nowcaster and a portfolio backtester with half-spread, per-fill slippage, and Almgren–Chriss temporary market impact. The Streamlit **Forecasting** tab accepts two CSV files for a monthly CPI nowcast; the **Stress Test** tab also contains an execution-aware portfolio backtest with adjustable friction assumptions. Its equity/drawdown and execution-cost charts are interactive Plotly figures with hover details and date-range controls.

### MIDAS input files

Daily observations CSV requires `date` plus one or more numeric market columns (for example `commodity_price,bond_yield`). Monthly CPI CSV requires `period,cpi,release_date`, where `release_date` is the actual public release timestamp. MIDAS training only uses daily observations strictly before each release, and a nowcast only uses observations strictly before its selected `as_of` cutoff. Upload enough released CPI periods (at least three) and daily history for the configured lookback window.

The backend endpoint is `POST /api/quant/midas-nowcast` (Analyst/Admin token required):

```json
{
  "daily_observations": [
    {"date": "2024-01-02T00:00:00Z", "commodity_price": 74.2, "bond_yield": 4.1}
  ],
  "monthly_cpi": {"2023-10": 307.6, "2023-11": 307.9, "2023-12": 308.4},
  "release_dates": {
    "2023-10": "2023-11-14T13:30:00Z",
    "2023-11": "2023-12-12T13:30:00Z",
    "2023-12": "2024-01-11T13:30:00Z"
  },
  "target_period": "2024-01",
  "as_of": "2024-01-20T23:59:59Z",
  "lookback_days": 60,
  "alpha": 1.0
}
```

### Backtest cost assumptions and reported metrics

`POST /api/quant/backtest` accepts `half_spread_bps`, `base_slippage_bps`, `impact_eta`, and optional per-asset `average_daily_volume` and `daily_slippage_bps`. The dashboard exposes the first three controls. Its simulator varies slippage with realized daily moves; market-data runs can supply per-fill slippage and ADV through the API. Results expose Sharpe, Sortino, Calmar, historical 95% expected shortfall (loss convention), maximum drawdown and its duration (business-day observations), annualized volatility, equity, and transaction costs. The Streamlit demo backtest uses synthetic daily returns; use the backend endpoint for warehouse-backed market data.

## 📄 License & Attribution

Designed and engineered for institutional macroeconomic research, quantitative analysis, and educational exploration. 
Data sourced via FRED (Federal Reserve Bank of St. Louis), World Bank Open Data, and public central bank releases.
