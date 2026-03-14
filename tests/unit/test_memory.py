import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import uuid

from app.memory.short_term import get_messages, append_messages, clear_session
from app.memory.long_term import LongTermMemory


class TestShortTermMemory:
    """Unit tests for short-term memory (Redis)."""

    @pytest.mark.asyncio
    @patch("app.memory.short_term.cache_get")
    async def test_get_messages_empty(self, mock_cache_get):
        """Test getting messages when none exist."""
        mock_cache_get.return_value = None
        result = await get_messages("test-session")
        assert result == []

    @pytest.mark.asyncio
    @patch("app.memory.short_term.cache_get")
    @patch("app.memory.short_term.cache_set")
    async def test_append_messages(self, mock_cache_set, mock_cache_get):
        """Test appending messages."""
        mock_cache_get.return_value = None
        await append_messages("test-session", "Hello", "Hi there")
        mock_cache_set.assert_called_once()


class TestLongTermMemory:
    """Unit tests for long-term memory (FAISS)."""

    @pytest.mark.asyncio
    @patch("app.memory.long_term.FAISS")
    @patch("app.memory.long_term._get_embeddings")
    async def test_add_memory(self, mock_embeddings, mock_faiss):
        """Test adding memory to FAISS."""
        mock_index = MagicMock()
        mock_faiss.from_texts.return_value = mock_index

        memory = LongTermMemory()
        user_id = uuid.uuid4()

        with patch.object(memory, "_get_or_create_index", return_value=mock_index):
            await memory.add_memory(user_id, "Test conversation")

        mock_index.add_texts.assert_called_once()
        mock_index.save_local.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.memory.long_term.FAISS")
    async def test_search_returns_context(self, mock_faiss):
        """Test semantic search returns context."""
        mock_index = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Previous conversation"
        mock_index.similarity_search.return_value = [mock_doc]

        memory = LongTermMemory()
        user_id = uuid.uuid4()

        with patch.object(memory, "_get_or_create_index", return_value=mock_index):
            result = await memory.search(user_id, "query", k=5)

        assert result == "Previous conversation"

    @pytest.mark.asyncio
    @patch("app.memory.long_term.FAISS")
    async def test_search_empty(self, mock_faiss):
        """Test search returns empty string when no results."""
        mock_index = MagicMock()
        mock_index.similarity_search.return_value = []

        memory = LongTermMemory()
        user_id = uuid.uuid4()

        with patch.object(memory, "_get_or_create_index", return_value=mock_index):
            result = await memory.search(user_id, "query", k=5)

        assert result == ""