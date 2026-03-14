from typing import Annotated

import httpx
from authlib.integrations.requests_client import OAuth2Session
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.dependencies import get_current_user
from app.db.repositories import user_repo
from app.schemas.auth import OAuthCallbackResponse, UserOut

from ...db.session import get_db

settings = get_settings()

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/google")
async def login_google(request: Request) -> dict:
    """Redirect to Google OAuth consent URL."""
    from fastapi import redirect

    oauth2_session = OAuth2Session(
        settings.google_client_id,
        settings.google_client_secret,
        scope=["openid", "email", "profile"],
    )

    uri, _ = oauth2_session.create_authorization_url(
        "https://accounts.google.com/o/oauth2/v2/auth",
        redirect_uri=settings.google_redirect_uri,
    )

    return redirect(str(uri))


@router.get("/google/callback")
async def google_callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> OAuthCallbackResponse:
    """Exchange code for token, fetch user info, create/update user."""
    code = request.query_params.get("code")

    if not code:
        raise ValueError("No code provided")

    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.google_redirect_uri,
            },
        )

        token_data = token_response.json()
        access_token = token_data["access_token"]

        userinfo_response = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        userinfo = userinfo_response.json()

    google_id = userinfo["id"]
    email = userinfo["email"]
    name = userinfo.get("name", "")
    avatar_url = userinfo.get("picture")

    user = await user_repo.upsert(
        db=db,
        google_id=google_id,
        email=email,
        name=name,
        avatar_url=avatar_url,
    )

    await db.commit()

    request.session["user_id"] = str(user.id)

    return OAuthCallbackResponse(
        user=UserOut.model_validate(user),
        message="Login successful",
    )


@router.post("/logout")
async def logout(request: Request) -> dict:
    """Clear session and log out."""
    request.session.clear()
    return {"message": "logged out"}


@router.get("/me")
async def get_me(
    current_user: Annotated[UserOut, Depends(get_current_user)],
) -> UserOut:
    """Get current user info."""
    return current_user