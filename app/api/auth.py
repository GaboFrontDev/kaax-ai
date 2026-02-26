from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.infra.settings import Settings, get_settings

security = HTTPBearer(auto_error=False)


async def require_bearer_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    settings: Settings = Depends(get_settings),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = credentials.credentials
    if token not in settings.api_tokens:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API token")

    return token
