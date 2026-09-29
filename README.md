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

### ✅ Enhanced Security & Authentication
- **RS256 JWT-scoped access** for macro research endpoints
- **Role-based access control** (Admin, Analyst, Viewer)
- **Automatic RSA key generation** for development environments
- **Streamlit-compatible security** functions for frontend integration

---

## 📄 License & Attribution

Designed and engineered for institutional macroeconomic research, quantitative analysis, and educational exploration. 
Data sourced via FRED (Federal Reserve Bank of St. Louis), World Bank Open Data, and public central bank releases.
