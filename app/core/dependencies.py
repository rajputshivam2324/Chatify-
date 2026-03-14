import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.redis_client import check_rate_limit
from app.core.config import get_settings
from app.db.repositories.user_repo import user_repo

from app.db.session import get_db
from app.schemas.auth import UserOut

settings = get_settings()


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> UserOut:
    """Reads user_id from Starlette session, fetches from DB."""
    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user session")

    user = await user_repo.get_by_id(db, user_uuid)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return UserOut.model_validate(user)


async def require_auth(
    current_user: Annotated[UserOut, Depends(get_current_user)],
) -> UserOut:
    """Alias for get_current_user, used in protected routes."""
    return current_user


async def rate_limit(
    request: Request,
    current_user: Annotated[UserOut, Depends(get_current_user)],
) -> None:
    """Check rate limit for the user."""
    await check_rate_limit(current_user.id, settings.rate_limit_per_minute)