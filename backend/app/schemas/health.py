"""Health (liveness) and readiness (model loaded) response schemas."""

from typing import List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    service: str


class ReadinessResponse(BaseModel):
    status: str                    # "ready" (>=1 model loaded) or "model_unavailable"
    model_loaded: bool
    loaded_directions: List[str]
    missing_directions: List[str]
    device: str
    gpu_available: bool
