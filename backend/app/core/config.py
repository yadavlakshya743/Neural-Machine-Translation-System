"""Application settings and environment configuration."""

import os
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Neural Machine Translation & Localization API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Server configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Model defaults
    MODEL_CHECKPOINT_PATH: str = "models/best_transformer_en_hi.pt"
    TOKENIZER_EN_PATH: str = "ml/tokenization/spm_en.model"
    TOKENIZER_HI_PATH: str = "ml/tokenization/spm_hi.model"
    DEVICE: str = "cpu"

    # Decoding Defaults
    DEFAULT_DECODING_STRATEGY: str = "beam_search"
    DEFAULT_BEAM_SIZE: int = 4
    DEFAULT_LENGTH_PENALTY: float = 0.6
    MAX_GEN_LENGTH: int = 128

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()
