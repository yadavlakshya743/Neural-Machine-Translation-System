"""Health and Readiness response schemas."""

from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    service: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    model_checkpoint: Optional[str] = None
    device: str
    gpu_available: bool
