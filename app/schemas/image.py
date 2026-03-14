from pydantic import BaseModel, Field


class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)


class ImageEditRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    image_data: str = Field(..., min_length=1)