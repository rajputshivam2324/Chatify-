import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.llm_service import invoke_llm, MODEL_CONFIGS, LLMInvocationError


class TestLLMService:
    """Unit tests for the LLM service."""

    def test_model_configs_exist(self):
        """Test that all required model configs are defined."""
        assert "llama" in MODEL_CONFIGS
        assert "qwen" in MODEL_CONFIGS
        assert "gemma" in MODEL_CONFIGS
        assert "deepseek" in MODEL_CONFIGS

    def test_llama_config(self):
        """Test LLaMA configuration."""
        config = MODEL_CONFIGS["llama"]
        assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.provider is None
        assert config.max_tokens == 10000

    def test_qwen_config(self):
        """Test Qwen configuration."""
        config = MODEL_CONFIGS["qwen"]
        assert config.model_id == "Qwen/Qwen2.5-VL-32B-Instruct"
        assert config.provider == "fireworks-ai"
        assert config.max_tokens == 25000

    @pytest.mark.asyncio
    async def test_invoke_llm_unknown_model(self):
        """Test that unknown model raises error."""
        with pytest.raises(LLMInvocationError):
            await invoke_llm("unknown_model", [{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    @patch("app.services.llm_service._get_client")
    async def test_invoke_llm_success(self, mock_get_client):
        """Test successful LLM invocation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await invoke_llm("llama", [{"role": "user", "content": "test"}])

        assert result == "Test response"
        mock_client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.llm_service._get_client")
    async def test_invoke_llm_empty_response(self, mock_get_client):
        """Test empty response handling."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMInvocationError):
            await invoke_llm("llama", [{"role": "user", "content": "test"}])