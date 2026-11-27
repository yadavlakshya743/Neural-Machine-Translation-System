"""FastAPI dependencies: authenticated user from a Bearer token."""

from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app import db
from backend.app.core.security import decode_access_token

bearer = HTTPBearer(auto_error=False)


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> Dict[str, Any]:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "UNAUTHORIZED", "message": "Login required (missing or invalid token)."},
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorized
    payload = decode_access_token(creds.credentials)
    if not payload or "sub" not in payload:
        raise unauthorized
    user = db.get_user_by_id(int(payload["sub"]))
    if user is None:
        raise unauthorized
    return user
