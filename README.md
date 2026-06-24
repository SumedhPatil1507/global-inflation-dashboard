# Global Inflation Dashboard

## Overview
A production‑grade FastAPI backend combined with a Streamlit frontend that visualizes macro‑economic indicators and market asset data. The system leverages:
- **PostgreSQL + TimescaleDB** for time‑series storage
- **Asyncpg** for async DB access
- **Celery + Redis** for background model training jobs
- **Redis** for heavy‑query caching
- **RS256 JWT** authentication with RSA keys
- **Immutable audit trail** for every API request
- **Streamlit app** for interactive visualisation

## Streamlit Frontend
The live dashboard is hosted at:
https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app/

You can run the Streamlit UI locally with:
```bash
streamlit run streamlit/app.py
```

## Architecture Diagram
*(Add architecture diagram image here if desired)*

## Environment Variables
| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | Asyncpg DSN for PostgreSQL | `postgresql+asyncpg://postgres:secure_vault_pass@postgres-db:5432/macro_analytics` |
| `RSA_PRIVATE_KEY_PATH` | Path to RSA private key for JWT signing | `certs/private_key.pem` |
| `RSA_PUBLIC_KEY_PATH` | Path to RSA public key for JWT verification | `certs/public_key.pem` |
| `REDIS_URL` | Redis instance for generic caching | `redis://redis-queue:6379/0` |
| `CELERY_BROKER_URL` | Celery broker (Redis) | `redis://redis-queue:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery result backend (Redis) | `redis://redis-queue:6379/1` |
| `REDIS_CACHE_URL` | Dedicated Redis cache for data‑heavy functions | `redis://redis-queue:6379/2` |
| `AUDIT_TABLE_NAME` | Table storing immutable audit logs | `security_audit_ledger` |
| `APP_ENV` | Application environment (`development`, `staging`, `production`) | `development` |

## Setup & Development
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd global-inflation-dashboard
   ```
2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
3. **Configure environment**
   - Copy `.env.example` to `.env` and fill in the values above.
   - Ensure RSA key files exist at the configured paths.
4. **Run database migrations**
   ```bash
   psql -f backend/migrations.sql
   ```
5. **Start services** (Docker Compose is recommended)
   ```bash
   docker-compose up -d
   ```
6. **Start the FastAPI server**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
7. **Start the Streamlit UI** (optional, for local dev)
   ```bash
   streamlit run streamlit/app.py
   ```

## API Endpoints
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/token` | Obtain JWT (RS256) |
| `POST` | `/api/ml/train` | Submit training job (Celery) |
| `POST` | `/api/ml/predict` | Stateless inference using latest serialized model |
| `POST` | `/api/quant/backtest` | Run backtest on supplied parameters |
| `GET` | `/health` | Health check |

## Audit Trail
All API requests are logged immutably in the `security_audit_ledger` table by `AuditMiddleware`. Logged fields include user, role, IP, endpoint, method, request payload, execution time, and a cryptographic request signature.

## Testing
```bash
pytest tests/    # runs the test suite
```

## Deployment
- **Docker image**: `docker build -t inflation-dashboard .`
- Push the image to your registry and deploy via your orchestrator of choice.

## Git Workflow
```bash
git add .
git commit -m "Refresh README, add Streamlit URL, and finalize backend updates"
git push origin main
```

---
*All components follow best‑practice security, observability, and performance guidelines.*
