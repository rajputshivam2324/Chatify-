import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import ChatResponse
from app.services import chat_service


async def handle_chat(
    user_id: uuid.UUID,
    session_id: str,
    model_key: str,
    user_message: str,
    db: AsyncSession,
) -> ChatResponse:
    """Shared chat handler for all models."""
    return await chat_service.process_message(
        user_id=user_id,
        session_id=session_id,
        model_key=model_key,
        user_message=user_message,
        db=db,
    )