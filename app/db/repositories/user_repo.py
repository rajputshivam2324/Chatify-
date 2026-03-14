import uuid
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.user import User


class UserRepositoryProtocol(Protocol):
    async def get_by_google_id(self, db: AsyncSession, google_id: str) -> User | None:
        ...

    async def get_by_id(self, db: AsyncSession, user_id: uuid.UUID) -> User | None:
        ...

    async def upsert(
        self,
        db: AsyncSession,
        google_id: str,
        email: str,
        name: str,
        avatar_url: str | None,
    ) -> User:
        ...


class UserRepository:
    async def get_by_google_id(self, db: AsyncSession, google_id: str) -> User | None:
        result = await db.execute(
            select(User).where(User.google_id == google_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, db: AsyncSession, user_id: uuid.UUID) -> User | None:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        db: AsyncSession,
        google_id: str,
        email: str,
        name: str,
        avatar_url: str | None,
    ) -> User:
        result = await db.execute(
            select(User).where(User.google_id == google_id)
        )
        user = result.scalar_one_or_none()

        if user:
            user.email = email
            user.name = name
            user.avatar_url = avatar_url
        else:
            user = User(
                google_id=google_id,
                email=email,
                name=name,
                avatar_url=avatar_url,
            )
            db.add(user)

        await db.flush()
        await db.refresh(user)
        return user


user_repo = UserRepository()