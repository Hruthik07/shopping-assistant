"""Pytest configuration and fixtures for tests."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables BEFORE any imports
os.environ["ANTHROPIC_API_KEY"] = "test-key-for-ci"
os.environ["OPENAI_API_KEY"] = "test-key-for-ci"
os.environ["SERPER_API_KEY"] = "test-key-for-ci"
os.environ["DATABASE_URL"] = "sqlite:///./test_shopping_assistant.db"
os.environ["CACHE_ENABLED"] = "false"
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["DEEPEVAL_ENABLED"] = "false"

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables for all tests."""
    # Ensure test environment is configured
    os.environ["ANTHROPIC_API_KEY"] = "test-key-for-ci"
    os.environ["OPENAI_API_KEY"] = "test-key-for-ci"
    os.environ["SERPER_API_KEY"] = "test-key-for-ci"
    os.environ["DATABASE_URL"] = "sqlite:///./test_shopping_assistant.db"
    os.environ["CACHE_ENABLED"] = "false"
    os.environ["LANGFUSE_ENABLED"] = "false"
    os.environ["DEEPEVAL_ENABLED"] = "false"
    yield
    # Cleanup if needed
    pass


@pytest.fixture
def mock_llm():
    """Mock LLM for tests that don't need real API calls."""
    with patch("src.agent.shopping_agent.ChatAnthropic") as mock_anthropic, \
         patch("src.agent.shopping_agent.ChatOpenAI") as mock_openai:
        mock_llm_instance = MagicMock()
        mock_anthropic.return_value = mock_llm_instance
        mock_openai.return_value = mock_llm_instance
        yield mock_llm_instance


@pytest.fixture
def mock_cache():
    """Mock cache service for tests with async method support."""
    with patch("src.utils.cache.cache_service") as mock:
        # Set basic attributes
        mock.enabled = False
        mock.redis_client = None
        
        # Configure all async methods to use AsyncMock
        mock.connect = AsyncMock(return_value=None)
        mock.disconnect = AsyncMock(return_value=None)
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=False)
        mock.delete = AsyncMock(return_value=False)
        mock.get_llm_response = AsyncMock(return_value=None)
        mock.set_llm_response = AsyncMock(return_value=False)
        mock.get_product_search = AsyncMock(return_value=None)
        mock.set_product_search = AsyncMock(return_value=False)
        mock.get_embedding = AsyncMock(return_value=None)
        mock.set_embedding = AsyncMock(return_value=False)
        mock.get_session_data = AsyncMock(return_value=None)
        mock.set_session_data = AsyncMock(return_value=False)
        mock.get_conversation_history = AsyncMock(return_value=None)
        mock.set_conversation_history = AsyncMock(return_value=False)
        mock.get_stats = AsyncMock(return_value={"hits": 0, "misses": 0, "sets": 0, "errors": 0})
        
        # Mock get_cache_stats (synchronous method)
        mock.get_cache_stats = MagicMock(return_value={"hit_rate": 0.0, "hits": 0, "misses": 0})
        
        yield mock
