from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    DATA_DIR: str = "./data"
    COMPANY_ID: str = "mysoftheaven"

    TOP_K: int = 8
    MMR_LAMBDA: float = 0.6
    MIN_SCORE: float = 0.30
    MAX_CONTEXT_CHARS: int = 12000
