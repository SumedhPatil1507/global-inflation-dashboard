"""Central config — reads from environment variables / .env file."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # JWT
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm:  str = "HS256"
    jwt_expire_min: int = 60

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""

    # FRED API
    fred_api_key: str = ""

    # Alpha Vantage (commodity prices)
    alpha_vantage_key: str = ""

    # App
    app_env: str = "development"
    allowed_origins: list[str] = ["http://localhost:8501",
                                   "https://global-inflation-dashboard-cmuugxnnh2kqffda2e78app.streamlit.app"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
