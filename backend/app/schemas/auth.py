"""Authentication request/response schemas."""

from pydantic import BaseModel, Field


class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    username: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: str
