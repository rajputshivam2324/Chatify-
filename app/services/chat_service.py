import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import message_repo, session_repo
from app.schemas.chat import ChatResponse
from app.services import session_service

from . import llm_service
from ..memory import graph


async def process_message(
    user_id: uuid.UUID,
    session_id: str,
    model_key: str,
    user_message: str,
    db: AsyncSession,
) -> ChatResponse:
    """
    Orchestrates: memory read -> LLM call -> memory write -> DB persist
    """
    db_session = await session_service.get_or_create_session(
        db=db,
        user_id=user_id,
        session_id=session_id,
        model_name=model_key,
    )

    reply = await graph.run_graph(
        user_id=user_id,
        session_id=session_id,
        model_key=model_key,
        user_message=user_message,
    )

    await message_repo.create(
        db=db,
        session_id=db_session.id,
        role="user",
        content=user_message,
    )
    await message_repo.create(
        db=db,
        session_id=db_session.id,
        role="assistant",
        content=reply,
    )

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        model=model_key,
    )