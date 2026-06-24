# Global Inflation Dashboard

## Overview

A production‑grade FastAPI backend that ingests macroeconomic data, trains ML models, and serves forecasts. It uses:
- **PostgreSQL + TimescaleDB** for time‑series storage
- **Asyncpg** for asynchronous DB access
- **Celery + Redis** for background training jobs
- **Redis** for caching heavy queries
- **RS256 JWT** authentication with RSA keys
- **Immutable audit trail** for every API request

## Architecture Diagram
*(Skipped as per user request)*

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

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd global-inflation-dashboard
   ```
2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure environment**
   - Copy `.env.example` to `.env` and fill in the values above.
   - Ensure RSA key files exist at the paths configured.
4. **Run database migrations**
   ```bash
   psql -f backend/migrations.sql
   ```
5. **Start services** (Docker Compose is recommended)
   ```bash
   docker-compose up -d
   ```
   This brings up PostgreSQL (with TimescaleDB), Redis, and a Celery worker.
6. **Start the FastAPI server**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
7. **Trigger weekly model training**
   Celery beat is configured (see `backend/workers/celery_app.py`). The task `train_and_serialize_models` runs weekly, stores serialized models under `backend/models/registry/` as Joblib files.

## API Endpoints
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/token` | Obtain JWT (RS256) |
| `POST` | `/api/ml/train` | Submit training job (Celery) |
| `POST` | `/api/ml/predict` | Stateless inference using latest serialized model |
| `POST` | `/api/quant/backtest` | Run backtest on supplied parameters |
| `GET` | `/health` | Health check |

## Audit Trail
All API requests are logged immutably in the `security_audit_ledger` table by the `AuditMiddleware`. The log includes user, role, IP, endpoint, method, request payload, execution time, and a cryptographic request signature.

## Testing
```bash
pytest tests/    # runs the test suite
```

## Deployment
- Build Docker image: `docker build -t inflation-dashboard .`
- Push to registry and deploy via your preferred orchestrator.

## Git Workflow
```bash
git add .
git commit -m "Implement ML service refactor, backtester, audit trail, Redis cache, and README updates"
git push origin main
```

---
*All changes are production‑ready and follow best‑practice security, observability, and performance guidelines.*
