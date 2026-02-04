"""Unit tests for Langfuse trace creation and error handling."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.agent.shopping_agent import ShoppingAgent
from src.analytics.langfuse_client import langfuse_client


class TestLangfuseTraceCreation:
    """Test Langfuse trace creation and error handling."""

    @pytest.fixture
    def agent(self):
        """Create a ShoppingAgent instance for testing."""
        return ShoppingAgent()

    @pytest.mark.asyncio
    async def test_trace_creation_success(self, agent):
        """Test successful Langfuse trace creation."""
        with patch.object(langfuse_client, "enabled", True):
            with patch.object(langfuse_client, "trace") as mock_trace:
                # Mock trace object with id attribute
                mock_trace_obj = Mock()
                mock_trace_obj.id = "test-trace-id-123"
                mock_trace.return_value = mock_trace_obj

                # Call process_query which creates trace
                query = "Find me headphones"
                session_id = "test-session"

                # Mock other dependencies
                with patch.object(
                    agent, "_get_conversation_history", new_callable=AsyncMock
                ) as mock_history:
                    with patch.object(
                        agent, "_get_user_preferences", new_callable=AsyncMock
                    ) as mock_prefs:
                        with patch.object(agent, "_detect_intent") as mock_intent:
                            with patch.object(agent, "_build_messages") as mock_messages:
                                with patch.object(agent, "_create_context_hash") as mock_hash:
                                    with patch.object(
                                        agent, "_process_with_llm", new_callable=AsyncMock
                                    ) as mock_llm:
                                        with patch.object(
                                            agent, "_store_conversation", new_callable=AsyncMock
                                        ):
                                            with patch.object(agent, "_track_analytics"):

                                                mock_history.return_value = []
                                                mock_prefs.return_value = {}
                                                mock_intent.return_value = {"type": "search"}
                                                mock_messages.return_value = []
                                                mock_hash.return_value = "test-hash"
                                                mock_llm.return_value = (
                                                    "Test response",
                                                    [],
                                                    ["search_products"],
                                                    None,
                                                )

                                                result = await agent.process_query(
                                                    query=query, session_id=session_id
                                                )

                                                # Verify trace was created
                                                assert mock_trace.called
                                                assert (
                                                    result.get("langfuse_trace_id")
                                                    == "test-trace-id-123"
                                                )

    @pytest.mark.asyncio
    async def test_trace_creation_failure_graceful(self, agent):
        """Test that trace creation failure doesn't break the query."""
        with patch.object(langfuse_client, "enabled", True):
            with patch.object(
                langfuse_client, "trace", side_effect=Exception("Trace creation failed")
            ):

                query = "Find me headphones"
                session_id = "test-session"

                # Mock other dependencies
                with patch.object(
                    agent, "_get_conversation_history", new_callable=AsyncMock
                ) as mock_history:
                    with patch.object(
                        agent, "_get_user_preferences", new_callable=AsyncMock
                    ) as mock_prefs:
                        with patch.object(agent, "_detect_intent") as mock_intent:
                            with patch.object(agent, "_build_messages") as mock_messages:
                                with patch.object(agent, "_create_context_hash") as mock_hash:
                                    with patch.object(
                                        agent, "_process_with_llm", new_callable=AsyncMock
                                    ) as mock_llm:
                                        with patch.object(
                                            agent, "_store_conversation", new_callable=AsyncMock
                                        ):
                                            with patch.object(agent, "_track_analytics"):

                                                mock_history.return_value = []
                                                mock_prefs.return_value = {}
                                                mock_intent.return_value = {"type": "search"}
                                                mock_messages.return_value = []
                                                mock_hash.return_value = "test-hash"
                                                mock_llm.return_value = (
                                                    "Test response",
                                                    [],
                                                    ["search_products"],
                                                    None,
                                                )

                                                # Should not raise exception
                                                result = await agent.process_query(
                                                    query=query, session_id=session_id
                                                )

                                                # Query should still succeed
                                                assert result is not None
                                                assert "response" in result
                                                # Trace ID should be None
                                                assert result.get("langfuse_trace_id") is None

    @pytest.mark.asyncio
    async def test_trace_id_extraction_with_dict(self, agent):
        """Test trace ID extraction when trace is a dict."""
        with patch.object(langfuse_client, "enabled", True):
            with patch.object(langfuse_client, "trace") as mock_trace:
                # Mock trace as dict
                mock_trace.return_value = {"id": "dict-trace-id-456"}

                query = "Find me headphones"
                session_id = "test-session"

                with patch.object(
                    agent, "_get_conversation_history", new_callable=AsyncMock
                ) as mock_history:
                    with patch.object(
                        agent, "_get_user_preferences", new_callable=AsyncMock
                    ) as mock_prefs:
                        with patch.object(agent, "_detect_intent") as mock_intent:
                            with patch.object(agent, "_build_messages") as mock_messages:
                                with patch.object(agent, "_create_context_hash") as mock_hash:
                                    with patch.object(
                                        agent, "_process_with_llm", new_callable=AsyncMock
                                    ) as mock_llm:
                                        with patch.object(
                                            agent, "_store_conversation", new_callable=AsyncMock
                                        ):
                                            with patch.object(agent, "_track_analytics"):

                                                mock_history.return_value = []
                                                mock_prefs.return_value = {}
                                                mock_intent.return_value = {"type": "search"}
                                                mock_messages.return_value = []
                                                mock_hash.return_value = "test-hash"
                                                mock_llm.return_value = (
                                                    "Test response",
                                                    [],
                                                    ["search_products"],
                                                    None,
                                                )

                                                result = await agent.process_query(
                                                    query=query, session_id=session_id
                                                )

                                                assert (
                                                    result.get("langfuse_trace_id")
                                                    == "dict-trace-id-456"
                                                )

    @pytest.mark.asyncio
    async def test_trace_id_extraction_with_trace_id_attribute(self, agent):
        """Test trace ID extraction when trace has trace_id attribute."""
        with patch.object(langfuse_client, "enabled", True):
            with patch.object(langfuse_client, "trace") as mock_trace:
                # Create a custom class that only has trace_id, not id
                class TraceWithTraceId:
                    def __init__(self):
                        self.trace_id = "trace-id-attr-789"

                mock_trace_obj = TraceWithTraceId()
                mock_trace.return_value = mock_trace_obj

                query = "Find me headphones"
                session_id = "test-session"

                with patch.object(
                    agent, "_get_conversation_history", new_callable=AsyncMock
                ) as mock_history:
                    with patch.object(
                        agent, "_get_user_preferences", new_callable=AsyncMock
                    ) as mock_prefs:
                        with patch.object(agent, "_detect_intent") as mock_intent:
                            with patch.object(agent, "_build_messages") as mock_messages:
                                with patch.object(agent, "_create_context_hash") as mock_hash:
                                    with patch.object(
                                        agent, "_process_with_llm", new_callable=AsyncMock
                                    ) as mock_llm:
                                        with patch.object(
                                            agent, "_store_conversation", new_callable=AsyncMock
                                        ):
                                            with patch.object(agent, "_track_analytics"):

                                                mock_history.return_value = []
                                                mock_prefs.return_value = {}
                                                mock_intent.return_value = {"type": "search"}
                                                mock_messages.return_value = []
                                                mock_hash.return_value = "test-hash"
                                                mock_llm.return_value = (
                                                    "Test response",
                                                    [],
                                                    ["search_products"],
                                                    None,
                                                )

                                                result = await agent.process_query(
                                                    query=query, session_id=session_id
                                                )

                                                assert (
                                                    result.get("langfuse_trace_id")
                                                    == "trace-id-attr-789"
                                                )

    @pytest.mark.asyncio
    async def test_trace_update_with_final_output(self, agent):
        """Test that trace is updated with final output."""
        with patch.object(langfuse_client, "enabled", True):
            with patch.object(langfuse_client, "trace") as mock_trace:
                with patch.object(langfuse_client, "update_trace") as mock_update:
                    with patch.object(langfuse_client, "flush") as mock_flush:

                        mock_trace_obj = Mock()
                        mock_trace_obj.id = "test-trace-id"
                        mock_trace.return_value = mock_trace_obj

                        query = "Find me headphones"
                        session_id = "test-session"

                        with patch.object(
                            agent, "_get_conversation_history", new_callable=AsyncMock
                        ) as mock_history:
                            with patch.object(
                                agent, "_get_user_preferences", new_callable=AsyncMock
                            ) as mock_prefs:
                                with patch.object(agent, "_detect_intent") as mock_intent:
                                    with patch.object(agent, "_build_messages") as mock_messages:
                                        with patch.object(
                                            agent, "_create_context_hash"
                                        ) as mock_hash:
                                            with patch.object(
                                                agent, "_process_with_llm", new_callable=AsyncMock
                                            ) as mock_llm:
                                                with patch.object(
                                                    agent,
                                                    "_store_conversation",
                                                    new_callable=AsyncMock,
                                                ):
                                                    with patch.object(agent, "_track_analytics"):

                                                        mock_history.return_value = []
                                                        mock_prefs.return_value = {}
                                                        mock_intent.return_value = {
                                                            "type": "search"
                                                        }
                                                        mock_messages.return_value = []
                                                        mock_hash.return_value = "test-hash"
                                                        mock_llm.return_value = (
                                                            "Test response",
                                                            [],
                                                            ["search_products"],
                                                            None,
                                                        )

                                                        await agent.process_query(
                                                            query=query, session_id=session_id
                                                        )

                                                        # Verify trace was updated
                                                        assert mock_update.called
                                                        update_call = mock_update.call_args
                                                        assert (
                                                            update_call[1]["trace_id"]
                                                            == "test-trace-id"
                                                        )
                                                        assert "output" in update_call[1]
                                                        assert "metadata" in update_call[1]
                                                        # Verify flush was called
                                                        assert mock_flush.called
