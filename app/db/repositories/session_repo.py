import uuid
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_session import ChatSession


class SessionRepositoryProtocol(Protocol):
    async def create(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        session_id: str,
        model_name: str,
        title: str | None,
    ) -> ChatSession:
        ...

    async def get_by_id(self, db: AsyncSession, session_id: uuid.UUID) -> ChatSession | None:
        ...

    async def get_by_session_id(
        self, db: AsyncSession, session_id: str, user_id: uuid.UUID
    ) -> ChatSession | None:
        ...

    async def list_for_user(
        self, db: AsyncSession, user_id: uuid.UUID, limit: int = 50
    ) -> list[ChatSession]:
        ...

    async def delete(self, db: AsyncSession, session_id: uuid.UUID) -> None:
        ...


class SessionRepository:
    async def create(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        session_id: str,
        model_name: str,
        title: str | None,
    ) -> ChatSession:
        session = ChatSession(
            user_id=user_id,
            session_id=session_id,
            model_name=model_name,
            title=title,
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)
        return session

    async def get_by_id(self, db: AsyncSession, db_session_id: uuid.UUID) -> ChatSession | None:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == db_session_id)
        )
        return result.scalar_one_or_none()

    async def get_by_session_id(
        self, db: AsyncSession, session_id: str, user_id: uuid.UUID
    ) -> ChatSession | None:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_user(
        self, db: AsyncSession, user_id: uuid.UUID, limit: int = 50
    ) -> list[ChatSession]:
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete(self, db: AsyncSession, db_session_id: uuid.UUID) -> None:
        session = await self.get_by_id(db, db_session_id)
        if session:
            await db.delete(session)
            await db.flush()


session_repo = SessionRepository()