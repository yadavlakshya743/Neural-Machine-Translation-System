"""Application settings, read from environment variables / .env (never hard-code secrets)."""

import secrets
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Neural Machine Translation & Localization API"
    VERSION: str = "1.1.0"
    API_V1_STR: str = "/api/v1"

    # Server configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Trained checkpoints, one per direction (written by scripts/train.py). The tokenizer
    # files saved next to each checkpoint are used automatically.
    MODEL_CHECKPOINT_PATH: str = "models/transformer_small_en-hi/best.pt"   # en -> hi
    MODEL_CHECKPOINT_PATH_HI_EN: str = "models/transformer_small_hi-en/best.pt"
    MODELS_DIR: str = "models"              # scanned for experiment/evaluation results
    DATA_REPORT_PATH: str = "data/processed/data_report.json"
    DEVICE: str = "cpu"                     # 'cuda' only if no training job is using the GPU

    # Decoding defaults and request bounds
    DEFAULT_DECODING_STRATEGY: str = "beam_search"
    DEFAULT_BEAM_SIZE: int = 4
    DEFAULT_LENGTH_PENALTY: float = 0.6
    MAX_GEN_LENGTH: int = 128

    # Authentication. A random key is generated if none is configured, which means tokens
    # stop working when the server restarts -- set SECRET_KEY in .env for real use.
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Persistence (SQLite file for users and translation history)
    DATABASE_URL: str = "sqlite:///./translation_history.db"

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="allow")

    @property
    def database_path(self) -> str:
        prefix = "sqlite:///"
        if not self.DATABASE_URL.startswith(prefix):
            raise ValueError("Only sqlite:/// DATABASE_URLs are supported.")
        return self.DATABASE_URL[len(prefix):]


settings = Settings()
