"""Unit tests for cache invalidation in conversation storage."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.agent.shopping_agent import ShoppingAgent
from src.memory.conversation_store import conversation_store
from src.utils.cache import cache_service


class TestCacheInvalidation:
    """Test cache invalidation after conversation storage."""

    @pytest.fixture
    def agent(self):
        """Create a ShoppingAgent instance for testing."""
        return ShoppingAgent()

    @pytest.mark.asyncio
    async def test_cache_invalidation_after_storage(self, agent):
        """Test that cache is invalidated after storing conversation."""
        session_id = "test-session-cache"
        query = "Find me headphones"
        agent_response = "Here are some headphones..."
        tools_used = ["search_products"]
        request_id = "test-request-123"

        with patch.object(conversation_store, "add_conversation") as mock_add:
            with patch.object(cache_service, "delete", new_callable=AsyncMock) as mock_delete:

                await agent._store_conversation(
                    session_id=session_id,
                    query=query,
                    agent_response=agent_response,
                    tools_used=tools_used,
                    request_id=request_id,
                )

                # Verify cache invalidation was called for common limits
                assert mock_delete.called
                delete_calls = [call[0][0] for call in mock_delete.call_args_list]

                # Check that all common limit values were invalidated
                expected_keys = [
                    f"conversation:{session_id}:5",
                    f"conversation:{session_id}:10",
                    f"conversation:{session_id}:20",
                ]

                for expected_key in expected_keys:
                    assert expected_key in delete_calls

    @pytest.mark.asyncio
    async def test_cache_invalidation_failure_doesnt_break_storage(self, agent):
        """Test that cache invalidation failure doesn't break conversation storage."""
        session_id = "test-session-error"
        query = "Find me headphones"
        agent_response = "Here are some headphones..."
        tools_used = ["search_products"]
        request_id = "test-request-456"

        with patch.object(conversation_store, "add_conversation") as mock_add:
            with patch.object(cache_service, "delete", new_callable=AsyncMock) as mock_delete:
                # Make cache deletion fail
                mock_delete.side_effect = Exception("Cache deletion failed")

                # Should not raise exception
                await agent._store_conversation(
                    session_id=session_id,
                    query=query,
                    agent_response=agent_response,
                    tools_used=tools_used,
                    request_id=request_id,
                )

                # Verify conversation was still stored
                assert mock_add.called

    @pytest.mark.asyncio
    async def test_cache_invalidation_called_with_correct_keys(self, agent):
        """Test that cache invalidation uses correct key format."""
        session_id = "test-session-keys"
        query = "Test query"
        agent_response = "Test response"
        tools_used = []
        request_id = "test-request-789"

        with patch.object(conversation_store, "add_conversation"):
            with patch.object(cache_service, "delete", new_callable=AsyncMock) as mock_delete:

                await agent._store_conversation(
                    session_id=session_id,
                    query=query,
                    agent_response=agent_response,
                    tools_used=tools_used,
                    request_id=request_id,
                )

                # Verify key format
                delete_calls = [call[0][0] for call in mock_delete.call_args_list]
                for call_key in delete_calls:
                    assert call_key.startswith(f"conversation:{session_id}:")
                    # Extract limit from key
                    limit = int(call_key.split(":")[-1])
                    assert limit in [5, 10, 20]
