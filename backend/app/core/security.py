"""Password hashing (PBKDF2-SHA256, Python standard library) and JWT access tokens."""

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

from backend.app.core.config import settings

ALGORITHM = "HS256"
PBKDF2_ITERATIONS = 310_000  # OWASP 2023 recommendation for PBKDF2-HMAC-SHA256


def hash_password(password: str) -> str:
    """Returns 'pbkdf2_sha256$<iterations>$<salt>$<hash>' (salted, slow by design)."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     base64.b64decode(salt_b64), int(iterations))
        return hmac.compare_digest(digest, base64.b64decode(hash_b64))  # constant-time compare
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Returns the payload, or None if the token is invalid or expired."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
