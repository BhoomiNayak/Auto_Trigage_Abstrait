"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq API
    groq_api_key: str = ""

    # Model selection
    groq_model_fast: str = "openai/gpt-oss-20b"
    groq_model_accurate: str = "openai/gpt-oss-20b"

    # App settings
    max_file_size_mb: int = 10
    max_retries: int = 3
    llm_timeout: int = 30  # seconds

    # Server
    app_name: str = "Auto-Triage & Document Extractor"
    app_version: str = "1.0.0"
    debug: bool = False

    @property
    def max_file_size_bytes(self) -> int:
        """Max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_configured(self) -> bool:
        """Check if the API key is properly configured."""
        return bool(self.groq_api_key) and self.groq_api_key != "gsk_your_api_key_here"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (loaded once, reused across requests)."""
    return Settings()
