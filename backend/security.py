import jwt
import secrets
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from backend.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_PLACEHOLDER_SECRETS = {"", "change-me-in-production", "changeme", "secret"}


def _require_jwt_secret() -> str:
    secret = settings.JWT_SECRET or ""
    if len(secret) < 32 or secret.lower() in _PLACEHOLDER_SECRETS:
        raise RuntimeError("JWT_SECRET must be a non-placeholder value of at least 32 characters")
    return secret


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload["type"] = "access"
    return jwt.encode(payload, _require_jwt_secret(), algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    payload["type"] = "refresh"
    return jwt.encode(payload, _require_jwt_secret(), algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    # PyJWT >= 2.10 enforces a string-typed "sub" claim. This codebase encodes
    # the int user id, so disable the verify_sub guard to stay compatible while
    # leaving signature + expiry checks active.
    return jwt.decode(
        token,
        _require_jwt_secret(),
        algorithms=[settings.JWT_ALGORITHM],
        options={"verify_sub": False},
    )


def generate_token() -> str:
    return secrets.token_urlsafe(32)
