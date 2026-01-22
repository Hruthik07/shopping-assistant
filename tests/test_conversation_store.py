"""Unit tests for conversation store async/await functionality."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.memory.conversation_store import ConversationStore, conversation_store


class TestConversationStoreAsync:
    """Test async/await functionality in ConversationStore."""
    
    @pytest.fixture
    def store(self):
        """Create a ConversationStore instance for testing."""
        return ConversationStore()
    
    @pytest.mark.asyncio
    async def test_get_context_for_llm_is_async(self, store):
        """Test that get_context_for_llm is properly async and awaits."""
        session_id = "test-session-123"
        
        # Mock get_conversation_history to return test data
        with patch.object(store, 'get_conversation_history', new_callable=AsyncMock) as mock_history:
            mock_history.return_value = [
                {
                    "user_message": "Find me headphones",
                    "agent_response": "Here are some headphones..."
                },
                {
                    "user_message": "What about wireless ones?",
                    "agent_response": "Here are wireless options..."
                }
            ]
            
            # Call the async method
            result = await store.get_context_for_llm(session_id, limit=5)
            
            # Verify it was awaited (not a coroutine)
            assert isinstance(result, str)
            assert "Find me headphones" in result
            assert "Here are some headphones..." in result
            assert "Here are wireless options..." in result
            
            # Verify get_conversation_history was called and awaited
            assert mock_history.called
            assert mock_history.return_value is not None
    
    @pytest.mark.asyncio
    async def test_get_context_for_llm_empty_history(self, store):
        """Test get_context_for_llm with empty history."""
        session_id = "test-session-empty"
        
        with patch.object(store, 'get_conversation_history', new_callable=AsyncMock) as mock_history:
            mock_history.return_value = []
            
            result = await store.get_context_for_llm(session_id, limit=5)
            
            assert result == ""
            assert mock_history.called
    
    @pytest.mark.asyncio
    async def test_get_context_for_llm_handles_errors(self, store):
        """Test that get_context_for_llm handles errors gracefully."""
        session_id = "test-session-error"
        
        with patch.object(store, 'get_conversation_history', new_callable=AsyncMock) as mock_history:
            mock_history.side_effect = Exception("Database error")
            
            # Should raise the exception (not return a coroutine)
            with pytest.raises(Exception, match="Database error"):
                await store.get_context_for_llm(session_id, limit=5)
    
    @pytest.mark.asyncio
    async def test_get_context_for_llm_formats_correctly(self, store):
        """Test that get_context_for_llm formats conversation history correctly."""
        session_id = "test-session-format"
        
        with patch.object(store, 'get_conversation_history', new_callable=AsyncMock) as mock_history:
            mock_history.return_value = [
                {
                    "user_message": "Hello",
                    "agent_response": "Hi there!"
                }
            ]
            
            result = await store.get_context_for_llm(session_id, limit=1)
            
            # Check formatting
            assert "User: Hello" in result
            assert "Assistant: Hi there!" in result
            assert result.count("User:") == 1
            assert result.count("Assistant:") == 1
    
    @pytest.mark.asyncio
    async def test_get_context_for_llm_respects_limit(self, store):
        """Test that get_context_for_llm respects the limit parameter."""
        session_id = "test-session-limit"
        
        with patch.object(store, 'get_conversation_history', new_callable=AsyncMock) as mock_history:
            mock_history.return_value = [
                {"user_message": f"Message {i}", "agent_response": f"Response {i}"}
                for i in range(10)
            ]
            
            result = await store.get_context_for_llm(session_id, limit=3)
            
            # Verify limit was passed to get_conversation_history
            assert mock_history.called
            call_args = mock_history.call_args
            assert call_args[0][0] == session_id
            assert call_args[1]["limit"] == 3
    
    @pytest.mark.asyncio
    async def test_global_conversation_store(self):
        """Test that the global conversation_store instance works correctly."""
        session_id = "test-global-session"
        
        with patch.object(conversation_store, 'get_conversation_history', new_callable=AsyncMock) as mock_history:
            mock_history.return_value = [
                {
                    "user_message": "Test message",
                    "agent_response": "Test response"
                }
            ]
            
            result = await conversation_store.get_context_for_llm(session_id)
            
            assert isinstance(result, str)
            assert "Test message" in result
            assert "Test response" in result

