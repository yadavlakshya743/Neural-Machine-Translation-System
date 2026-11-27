"""Pydantic schemas: request validation (bounded inputs) and response contracts."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TranslationRequest(BaseModel):
    source_text: str = Field(
        ..., min_length=1, max_length=1000,
        description="Source text to translate (1-1000 characters).",
        json_schema_extra={"example": "I am going to college."},
    )
    source_lang: Literal["en", "hi"] = Field("en", description="Source language code.")
    target_lang: Literal["en", "hi"] = Field("hi", description="Target language code.")
    decoding_strategy: Literal["greedy", "beam_search"] = Field("beam_search")
    beam_size: int = Field(4, ge=1, le=10, description="Beam width (beam search only).")
    length_penalty: float = Field(0.6, ge=0.0, le=2.0, description="Length-penalty alpha.")
    max_length: int = Field(128, ge=10, le=256, description="Maximum tokens to generate.")

    @field_validator("source_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_text must contain non-whitespace characters")
        return v


class TranslationResponse(BaseModel):
    request_id: str
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_version: str
    decoding_strategy: str
    beam_size: Optional[int] = None
    latency_ms: float
    tokens_generated: int
    created_at: Optional[str] = None


class TranslationHistoryPage(BaseModel):
    total: int
    items: List[TranslationResponse]


class ModelInfo(BaseModel):
    direction: str
    run_name: str
    architecture: str = "Transformer encoder-decoder (Vaswani et al., 2017)"
    parameters: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    n_encoder_layers: Optional[int] = None
    n_decoder_layers: Optional[int] = None
    d_ff: Optional[int] = None
    src_vocab_size: Optional[int] = None
    tgt_vocab_size: Optional[int] = None
    tokenizer_sha256: Dict[str, Optional[str]] = {}
    train_pairs: Optional[int] = None
    checkpoint: str
    trained_epochs: Optional[int] = None
    best_dev_bleu: Optional[float] = None
    dataset: str = "IIT Bombay English-Hindi Parallel Corpus (official split)"
    device: str


class ModelsResponse(BaseModel):
    loaded: List[ModelInfo]
    missing_directions: List[str]


# Backwards-compatible summary used by the original /models/info endpoint.
class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    grounding: str
    source_languages: List[str]
    target_languages: List[str]
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    device: str
    dataset: str
    metrics: Dict[str, Any]
