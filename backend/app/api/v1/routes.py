"""API v1 router definitions for translation, health, readiness, and model metadata."""

import os
from typing import List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.health import HealthResponse, ReadinessResponse
from backend.app.schemas.translation import (
    TranslationRequest,
    TranslationResponse,
    ModelInfoResponse,
    HistoryItem,
)
from backend.app.services.translation_service import translation_service
from backend.app.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Liveness Probe",
)
async def health_check():
    """Returns liveness status of the translation service."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        service=settings.PROJECT_NAME,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["System"],
    summary="Model Readiness Probe",
)
async def readiness_check():
    """Checks whether the Transformer NMT model and tokenizers are loaded and ready."""
    gpu_avail = torch.cuda.is_available() if TORCH_AVAILABLE else False
    device = "cuda" if gpu_avail and settings.DEVICE == "cuda" else "cpu"

    return ReadinessResponse(
        status="ready" if translation_service.is_ready else "initializing",
        model_loaded=translation_service.is_ready,
        model_checkpoint=settings.MODEL_CHECKPOINT_PATH if os.path.exists(settings.MODEL_CHECKPOINT_PATH) else None,
        device=device,
        gpu_available=gpu_avail,
    )


@router.get(
    "/models/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get Model Architecture and Metadata",
)
async def get_model_info():
    """Returns metadata about the active Transformer model architecture, grounding, and hyperparameters."""
    return translation_service.get_model_info()


@router.post(
    "/translations",
    response_model=TranslationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Translation"],
    summary="Translate text between English and Hindi",
)
async def translate_text(request: TranslationRequest):
    """Generates translation using the custom Transformer Encoder-Decoder model.

    Supports configurable decoding strategy (greedy or beam search).
    """
    valid_directions = {("en", "hi"), ("hi", "en")}
    if (request.source_lang, request.target_lang) not in valid_directions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_LANGUAGE_DIRECTION",
                    "message": f"Language direction '{request.source_lang}->{request.target_lang}' is not supported. Supported: 'en<->hi'.",
                }
            },
        )

    try:
        response = translation_service.translate(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INFERENCE_FAILED",
                    "message": f"Translation inference failed: {str(e)}",
                }
            },
        )


@router.get(
    "/history",
    response_model=List[HistoryItem],
    tags=["Translation"],
    summary="Get Translation History",
)
async def get_translation_history():
    """Returns recent translation queries, directions, and latency statistics."""
    return translation_service.history
