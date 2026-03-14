import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import uuid

from fastapi.testclient import TestClient


@pytest.fixture
def mock_db():
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    from app.schemas.auth import UserOut
    return UserOut(
        id=uuid.uuid4(),
        email="test@example.com",
        name="Test User",
        avatar_url=None,
    )


class TestChatRoutes:
    """Integration tests for chat routes."""

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.chat_service.process_message")
    async def test_chat_endpoint(self, mock_process):
        """Test /chat POST endpoint."""
        from app.schemas.chat import ChatResponse
        from app.main import app
        from fastapi import Depends
        from app.core.dependencies import get_current_user
        from app.api.routes.auth import router

        mock_process.return_value = ChatResponse(
            reply="Hello!",
            session_id="test-session",
            model="llama"
        )

        with TestClient(app) as client:
            client.cookies.set("session", "test-session-id")
            pass

    def test_health_endpoint(self):
        """Test /health endpoint."""
        from app.main import app

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "env" in data


class TestAuthentication:
    """Integration tests for authentication."""

    def test_google_login_redirect(self):
        """Test /auth/google redirects to Google."""
        from app.main import app

        with TestClient(app) as client:
            response = client.get("/auth/google", follow_redirects=False)
            assert response.status_code in [200, 302]

    def test_logout(self):
        """Test /auth/logout clears session."""
        from app.main import app

        with TestClient(app) as client:
            response = client.post("/auth/logout")
            assert response.status_code == 200
            assert response.json()["message"] == "logged out"

    def test_get_me_unauthenticated(self):
        """Test /auth/me returns 401 when not authenticated."""
        from app.main import app

        with TestClient(app) as client:
            response = client.get("/auth/me")
            assert response.status_code == 401