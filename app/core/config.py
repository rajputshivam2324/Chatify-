from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_env: str = "production"
    app_secret: str = Field(default="")
    frontend_url: str = "http://localhost:3000"
    database_url: str = Field(default="")
    redis_url: str = Field(default="")
    session_ttl_seconds: int = 86400
    rate_limit_per_minute: int = 30
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    google_redirect_uri: str = Field(default="")
    hf_token: str = Field(default="")


@lru_cache
def get_settings() -> Settings:
    return Settings()