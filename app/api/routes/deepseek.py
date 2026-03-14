from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import chat as chat_routes
from app.core.dependencies import rate_limit, require_auth
from app.db.repositories import message_repo, session_repo
from app.schemas.auth import UserOut
from app.schemas.chat import ChatRequest, ChatResponse, HistoryResponse, MessageOut

from ...db.session import get_db

router = APIRouter(prefix="/deepseek", tags=["deepseek"])


@router.post("", response_model=ChatResponse)
async def send_message(
    body: ChatRequest,
    user: Annotated[UserOut, Depends(require_auth)],
    _: None = Depends(rate_limit),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """DeepSeek chat endpoint."""
    return await chat_routes.handle_chat(
        user_id=user.id,
        session_id=body.session_id,
        model_key="deepseek",
        user_message=body.user_message,
        db=db,
    )


@router.get("/{session_id}", response_model=HistoryResponse)
async def get_history(
    session_id: str,
    user: Annotated[UserOut, Depends(require_auth)],
    db: AsyncSession = Depends(get_db),
) -> HistoryResponse:
    """Get chat history for a session."""
    db_session = await session_repo.get_by_session_id(db, session_id, user.id)

    if not db_session:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Session not found")

    messages = await message_repo.list_for_session(db, db_session.id)

    return HistoryResponse(
        messages=[MessageOut.model_validate(m) for m in messages],
        session_id=session_id,
        model=db_session.model_name,
    )