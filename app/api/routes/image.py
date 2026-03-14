from typing import Annotated

import base64
import binascii

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from app.core.dependencies import require_auth
from app.schemas.auth import UserOut
from app.schemas.image import ImageEditRequest, ImageGenRequest
from app.services import image_service

router = APIRouter(prefix="/image", tags=["image"])


@router.post("/generate-image")
async def generate_image(
    body: ImageGenRequest,
    user: Annotated[UserOut, Depends(require_auth)],
) -> Response:
    """Generate an image from text prompt."""
    png_bytes = await image_service.generate_image(body.prompt)

    return Response(
        content=png_bytes,
        media_type="image/png",
    )


@router.post("/Chrono-Edit")
async def chrono_edit(
    body: ImageEditRequest,
    user: Annotated[UserOut, Depends(require_auth)],
) -> Response:
    """Edit an image using ChronoEdit."""
    try:
        image_bytes = base64.b64decode(body.image_data)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    result = await image_service.edit_image(image_bytes, body.prompt)

    return Response(
        content=result,
        media_type="image/png",
    )