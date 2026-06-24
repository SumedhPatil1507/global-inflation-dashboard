import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # JWT settings (RS256)
    jwt_algorithm: str = "RS256"
    jwt_expire_min: int = 60
    rsa_private_key_path: str = "certs/private_key.pem"
    rsa_public_key_path: str = "certs/public_key.pem"

    # Database URL (asyncpg)
    database_url: str = "postgresql+asyncpg://postgres:secure_vault_pass@postgres-db:5432/macro_analytics"

    # Redis / Celery configuration
    redis_url: str = "redis://redis-queue:6379/0"
    celery_broker_url: str = "redis://redis-queue:6379/0"
    celery_result_backend: str = "redis://redis-queue:6379/1"
    redis_cache_url: str = "redis://redis-queue:6379/2"

    # Audit trail table name
    audit_table_name: str = "security_audit_ledger"

    # Optional Supabase configuration
    supabase_url: str = ""
    supabase_key: str = ""

    # External API keys
    fred_api_key: str = ""
    alpha_vantage_key: str = ""

    # Application environment
    app_env: str = "development"
    allowed_origins: list[str] = [
        "http://localhost:8501",
        "https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()



