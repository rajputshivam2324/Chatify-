import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    timestamp: datetime


class HistoryResponse(BaseModel):
    messages: list[MessageOut]
    session_id: str
    model: str