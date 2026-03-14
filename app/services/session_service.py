import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_session import ChatSession
from app.db.repositories import session_repo


async def get_or_create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: str,
    model_name: str,
) -> ChatSession:
    """Get existing session or create a new one."""
    existing = await session_repo.get_by_session_id(db, session_id, user_id)

    if existing:
        return existing

    new_session = await session_repo.create(
        db=db,
        user_id=user_id,
        session_id=session_id,
        model_name=model_name,
        title=None,
    )
    return new_session


async def create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: str,
    model_name: str,
    title: str | None = None,
) -> ChatSession:
    """Create a new session."""
    return await session_repo.create(
        db=db,
        user_id=user_id,
        session_id=session_id,
        model_name=model_name,
        title=title,
    )


async def get_session(
    db: AsyncSession,
    session_id: str,
    user_id: uuid.UUID,
) -> ChatSession | None:
    """Get session by session_id and user_id."""
    return await session_repo.get_by_session_id(db, session_id, user_id)