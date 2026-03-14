import uuid
from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.message import Message


class MessageRepositoryProtocol(Protocol):
    async def create(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        role: str,
        content: str,
    ) -> Message:
        ...

    async def list_for_session(
        self, db: AsyncSession, session_id: uuid.UUID, limit: int = 100
    ) -> list[Message]:
        ...

    async def delete_for_session(self, db: AsyncSession, session_id: uuid.UUID) -> None:
        ...


class MessageRepository:
    async def create(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        role: str,
        content: str,
    ) -> Message:
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
        )
        db.add(message)
        await db.flush()
        await db.refresh(message)
        return message

    async def list_for_session(
        self, db: AsyncSession, session_id: uuid.UUID, limit: int = 100
    ) -> list[Message]:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.timestamp.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_for_session(self, db: AsyncSession, session_id: uuid.UUID) -> None:
        await db.execute(
            delete(Message).where(Message.session_id == session_id)
        )
        await db.flush()


message_repo = MessageRepository()