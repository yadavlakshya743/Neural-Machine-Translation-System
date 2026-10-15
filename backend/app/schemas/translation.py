"""Pydantic schemas for translation requests, responses, and model metadata."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class TranslationRequest(BaseModel):
    source_text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Source sentence or paragraph to be translated.",
        json_schema_extra={"example": "Machine learning models learn representations directly from data."}
    )
    source_lang: str = Field(
        default="en",
        description="Source language code ('en' or 'hi').",
        json_schema_extra={"example": "en"}
    )
    target_lang: str = Field(
        default="hi",
        description="Target language code ('hi' or 'en').",
        json_schema_extra={"example": "hi"}
    )
    decoding_strategy: str = Field(
        default="beam_search",
        description="Decoding strategy: 'greedy' or 'beam_search'.",
        json_schema_extra={"example": "beam_search"}
    )
    beam_size: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Beam width for beam search decoding.",
        json_schema_extra={"example": 4}
    )
    length_penalty: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Alpha parameter for length penalty.",
        json_schema_extra={"example": 0.6}
    )
    max_length: int = Field(
        default=128,
        ge=10,
        le=512,
        description="Maximum tokens to generate.",
        json_schema_extra={"example": 128}
    )


class TranslationResponse(BaseModel):
    request_id: str
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_version: str
    decoding_strategy: str
    latency_ms: float
    tokens_generated: int


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


class HistoryItem(BaseModel):
    id: str
    timestamp: str
    source_text: str
    translated_text: str
    direction: str
    strategy: str
    latency_ms: float
