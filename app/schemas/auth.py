import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    name: str
    avatar_url: str | None
    created_at: datetime


class OAuthCallbackResponse(BaseModel):
    user: UserOut
    message: str = "Login successful"