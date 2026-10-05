"""HTTP Bearer API-key verification for REST and WebSocket clients."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)


def _keys_match(provided: str, expected: str) -> bool:
    """Constant-time-ish comparison for the configured API key."""
    if len(provided) != len(expected):
        return False
    result = 0
    for left, right in zip(provided.encode("utf-8"), expected.encode("utf-8")):
        result |= left ^ right
    return result == 0


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    """Validate `Authorization: Bearer <API_KEY>` on REST routes.

    Raises:
        HTTPException: 401 when the header is missing or the token is invalid.
    """
    expected = settings.api_key.get_secret_value()
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not _keys_match(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def verify_ws_token(
    token: str | None = Query(default=None, description="Bearer API key."),
    settings: Settings = Depends(get_settings),
) -> str:
    """Validate the WebSocket handshake query parameter `?token=`.

    Browsers cannot set Authorization headers on the WS upgrade, so the
    same API key is passed as a query parameter.
    """
    expected = settings.api_key.get_secret_value()
    if not token or not _keys_match(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing WebSocket token.",
        )
    return token
