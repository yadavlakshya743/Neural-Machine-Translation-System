"""API v1: auth, translation + history, models, evaluations, health/readiness."""

import logging
import sqlite3
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from backend.app import db
from backend.app.api.deps import get_current_user
from backend.app.core.config import settings
from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.schemas.auth import Credentials, TokenResponse, UserResponse
from backend.app.schemas.health import HealthResponse, ReadinessResponse
from backend.app.schemas.translation import (ModelInfo, ModelInfoResponse, ModelsResponse,
                                             TranslationHistoryPage, TranslationRequest,
                                             TranslationResponse)
from backend.app.services import evaluation_service
from backend.app.services.translation_service import (DIRECTIONS, ModelNotAvailableError,
                                                      translation_service)

router = APIRouter()
logger = logging.getLogger("nmt.api")


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


# ------------------------------------------------------------------ system
@router.get("/health", response_model=HealthResponse, tags=["System"], summary="Liveness probe")
def health_check():
    """The API process is up (says nothing about the model)."""
    return HealthResponse(status="healthy", version=settings.VERSION, service=settings.PROJECT_NAME)


@router.get("/ready", response_model=ReadinessResponse, tags=["System"], summary="Model readiness probe")
def readiness_check(response: Response):
    """Ready only when at least one trained model is loaded (503 otherwise)."""
    import torch

    loaded = sorted(translation_service.translators)
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if loaded else "model_unavailable",
        model_loaded=bool(loaded),
        loaded_directions=loaded,
        missing_directions=[d for d in DIRECTIONS if d not in loaded],
        load_errors={d: msg[:300] for d, msg in translation_service.load_errors.items()},
        device=translation_service.device,
        gpu_available=torch.cuda.is_available(),
    )


# -------------------------------------------------------------------- auth
@router.post("/auth/register", response_model=UserResponse, status_code=201, tags=["Auth"])
def register(body: Credentials):
    try:
        user = db.create_user(body.username, hash_password(body.password))
    except sqlite3.IntegrityError:
        raise _error(409, "USERNAME_TAKEN", "That username is already registered.")
    logger.info("user_registered user_id=%s", user["id"])
    return UserResponse(**{k: user[k] for k in ("id", "username", "role", "created_at")})


@router.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(body: Credentials):
    user = db.get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["password_hash"]):
        logger.info("login_failed")
        raise _error(401, "INVALID_CREDENTIALS", "Incorrect username or password.")
    logger.info("login_ok user_id=%s", user["id"])
    return TokenResponse(access_token=create_access_token(user["id"], user["username"]),
                         expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
                         username=user["username"])


@router.post("/auth/logout", status_code=204, tags=["Auth"])
def logout(user: Dict[str, Any] = Depends(get_current_user)):
    """Tokens are stateless JWTs: the client discards its token. Kept for API completeness."""
    return Response(status_code=204)


@router.get("/auth/me", response_model=UserResponse, tags=["Auth"])
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return UserResponse(**{k: user[k] for k in ("id", "username", "role", "created_at")})


# ------------------------------------------------------------- translation
def _to_response(rec: Dict[str, Any]) -> TranslationResponse:
    return TranslationResponse(
        request_id=rec["id"], source_text=rec["source_text"], translated_text=rec["translated_text"],
        source_lang=rec["source_lang"], target_lang=rec["target_lang"],
        model_version=rec["model_version"], decoding_strategy=rec["decoding_strategy"],
        beam_size=rec.get("beam_size"), latency_ms=rec["latency_ms"],
        tokens_generated=rec["tokens_generated"], created_at=rec.get("created_at"),
    )


@router.post("/translations", response_model=TranslationResponse, tags=["Translation"],
             summary="Translate text (English <-> Hindi) and save it to your history")
def translate_text(request: TranslationRequest, user: Dict[str, Any] = Depends(get_current_user)):
    if request.source_lang == request.target_lang:
        raise _error(400, "INVALID_LANGUAGE_DIRECTION",
                     f"'{request.source_lang}->{request.target_lang}' is not supported. Use en->hi or hi->en.")
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    try:
        result = translation_service.translate(
            text=request.source_text, source_lang=request.source_lang, target_lang=request.target_lang,
            strategy=request.decoding_strategy, beam_size=request.beam_size,
            alpha=request.length_penalty, max_len=request.max_length,
        )
    except ModelNotAvailableError as e:
        raise _error(503, "MODEL_NOT_AVAILABLE", str(e))
    except Exception:
        # Full details go to the server log only; the client never sees a stack trace.
        logger.exception("inference_failed request_id=%s", req_id)
        raise _error(500, "INFERENCE_FAILED", "Translation could not be generated.")

    record = {
        "id": req_id, "user_id": user["id"],
        "source_lang": request.source_lang, "target_lang": request.target_lang,
        "source_text": request.source_text, "translated_text": result["translated_text"],
        "decoding_strategy": request.decoding_strategy,
        "beam_size": request.beam_size if request.decoding_strategy == "beam_search" else None,
        "model_version": result["model_version"], "latency_ms": result["latency_ms"],
        "tokens_generated": result["tokens_generated"],
    }
    db.insert_translation(record)
    # Log metadata only -- never the user's text (PRD security requirement).
    logger.info("translation_ok request_id=%s user_id=%s direction=%s strategy=%s latency_ms=%s tokens=%s",
                req_id, user["id"], f"{request.source_lang}-{request.target_lang}",
                request.decoding_strategy, result["latency_ms"], result["tokens_generated"])
    return _to_response(db.get_translation(user["id"], req_id))


@router.get("/translations", response_model=TranslationHistoryPage, tags=["Translation"],
            summary="Your translation history (newest first)")
def list_history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                 user: Dict[str, Any] = Depends(get_current_user)):
    return TranslationHistoryPage(
        total=db.count_translations(user["id"]),
        items=[_to_response(r) for r in db.list_translations(user["id"], limit, offset)],
    )


@router.get("/translations/{translation_id}", response_model=TranslationResponse, tags=["Translation"])
def get_history_item(translation_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    rec = db.get_translation(user["id"], translation_id)
    if rec is None:  # also returned for other users' records, so ids cannot be probed
        raise _error(404, "NOT_FOUND", "Translation not found.")
    return _to_response(rec)


# ------------------------------------------------------- models / evaluation
@router.get("/models/active", response_model=ModelsResponse, tags=["Model"],
            summary="Loaded model versions, architecture, tokenizer hashes")
def active_models():
    loaded = [ModelInfo(**translation_service.model_info(d)) for d in sorted(translation_service.translators)]
    return ModelsResponse(loaded=loaded,
                          missing_directions=[d for d in DIRECTIONS if d not in translation_service.translators])


@router.get("/models/info", response_model=ModelInfoResponse, tags=["Model"],
            summary="Summary of the active model (legacy endpoint)")
def get_model_info():
    if not translation_service.translators:
        raise _error(503, "MODEL_NOT_AVAILABLE", "No trained model is loaded.")
    info = translation_service.model_info(sorted(translation_service.translators)[0])
    return ModelInfoResponse(
        model_name=info["run_name"], architecture="Encoder-Decoder Attention Network",
        grounding="Vaswani et al. (NeurIPS 2017) 'Attention Is All You Need'",
        source_languages=[d.split("-")[0] for d in sorted(translation_service.translators)],
        target_languages=[d.split("-")[1] for d in sorted(translation_service.translators)],
        d_model=info["d_model"], n_heads=info["n_heads"],
        n_encoder_layers=info["n_encoder_layers"], n_decoder_layers=info["n_decoder_layers"],
        device=info["device"], dataset="IIT Bombay English-Hindi Parallel Corpus",
        metrics={"best_dev_bleu": info.get("best_dev_bleu")},
    )


@router.get("/evaluations", tags=["Evaluation"],
            summary="Experiments: training curves, dev/test BLEU and chrF++, examples")
def evaluations() -> Dict[str, Any]:
    return {"experiments": evaluation_service.list_experiments(),
            "data_report": evaluation_service.data_report()}
