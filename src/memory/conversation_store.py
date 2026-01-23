"""Store and retrieve conversation history."""

import time
import json as json_module
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from src.database.models import Conversation, Session as SessionModel
from src.database.db import SessionLocal
from src.analytics.logger import logger
from src.utils.cache import cache_service
from src.utils.config import settings
from src.utils.debug_log import file_debug_log


# #region debug instrumentation
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    try:
        file_debug_log(location, message, data, hypothesis_id=hypothesis_id)
    except Exception:
        pass


# #endregion


class ConversationStore:
    """Manage conversation history."""

    def add_conversation(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        tools_used: Optional[List[str]] = None,
        db: Optional[Session] = None,
    ):
        """Add a conversation to history."""
        # #region debug instrumentation
        _debug_log(
            "conversation_store.py:14",
            "add_conversation entry",
            {
                "session_id": session_id,
                "user_message_len": len(user_message),
                "agent_response_len": len(agent_response),
                "has_db": db is not None,
            },
            "A",
        )
        # #endregion
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            # Get session
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:30",
                "Session query result",
                {"session_found": session is not None, "session_id": session_id},
                "A",
            )
            # #endregion

            if not session:
                # Create session if it doesn't exist
                logger.info(f"Session not found, creating new session: {session_id}")
                # #region debug instrumentation
                _debug_log(
                    "conversation_store.py:34",
                    "Session not found - creating new session",
                    {"session_id": session_id},
                    "A",
                )
                # #endregion
                from src.memory.session_manager import session_manager

                # Ensure session exists in database (reuse existing db connection)
                try:
                    created_session_id = session_manager.get_or_create_session(
                        session_id=session_id, user_id=None, db=db
                    )
                    if created_session_id != session_id:
                        logger.warning(
                            f"Session ID mismatch: expected {session_id}, got {created_session_id}"
                        )
                    # Re-query for the session
                    session = (
                        db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
                    )
                    if not session:
                        logger.error(f"Failed to create session: {session_id}")
                        return
                except Exception as create_err:
                    logger.error(f"Error creating session: {create_err}")
                    return

            conversation = Conversation(
                session_id=session.id,
                user_message=user_message,
                agent_response=agent_response,
                tools_used=tools_used or [],
            )
            db.add(conversation)
            db.commit()
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:45",
                "Conversation added successfully",
                {
                    "session_id": session_id,
                    "conversation_id": conversation.id if hasattr(conversation, "id") else "N/A",
                },
                "A",
            )
            # #endregion

            logger.debug(f"Added conversation to session: {session_id}")
        except Exception as e:
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:48",
                "add_conversation exception",
                {"error": str(e), "error_type": type(e).__name__},
                "A",
            )
            # #endregion
            raise
        finally:
            if should_close:
                db.close()

    async def get_conversation_history(
        self, session_id: str, limit: int = 10, db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session (with caching)."""
        # #region debug instrumentation
        _debug_log(
            "conversation_store.py:52",
            "get_conversation_history entry",
            {"session_id": session_id, "limit": limit, "has_db": db is not None},
            "E",
        )
        # #endregion
        # Check cache first
        try:
            cached_history = await cache_service.get_conversation_history(session_id, limit=limit)
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:60",
                "Cache check result",
                {
                    "has_cached_history": cached_history is not None,
                    "cached_count": len(cached_history) if cached_history else 0,
                },
                "E",
            )
            # #endregion
            if cached_history is not None:
                logger.debug(f"Cache hit for conversation history: {session_id}")
                return cached_history
        except Exception as cache_error:
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:65",
                "Cache check failed",
                {"error": str(cache_error), "error_type": type(cache_error).__name__},
                "E",
            )
            # #endregion
            # Continue to database query

        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:71",
                "Session query in get_history",
                {"session_found": session is not None},
                "E",
            )
            # #endregion

            if not session:
                # #region debug instrumentation
                _debug_log(
                    "conversation_store.py:75",
                    "No session found - returning empty",
                    {"session_id": session_id},
                    "E",
                )
                # #endregion
                return []

            conversations = (
                db.query(Conversation)
                .filter(Conversation.session_id == session.id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
                .all()
            )
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:78",
                "Conversations queried",
                {"conversations_count": len(conversations)},
                "E",
            )
            # #endregion

            history = [
                {
                    "user_message": conv.user_message,
                    "agent_response": conv.agent_response,
                    "tools_used": conv.tools_used,
                    "created_at": conv.created_at.isoformat(),
                }
                for conv in reversed(conversations)
            ]

            # Cache the history
            try:
                await cache_service.set_conversation_history(
                    session_id=session_id,
                    history=history,
                    limit=limit,
                    ttl=settings.cache_session_ttl,
                )
                # #region debug instrumentation
                _debug_log(
                    "conversation_store.py:93",
                    "History cached successfully",
                    {"history_count": len(history)},
                    "E",
                )
                # #endregion
            except Exception as cache_set_error:
                # #region debug instrumentation
                _debug_log(
                    "conversation_store.py:97",
                    "Cache set failed",
                    {"error": str(cache_set_error), "error_type": type(cache_set_error).__name__},
                    "E",
                )
                # #endregion
                # Continue even if caching fails

            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:100",
                "get_conversation_history returning",
                {"history_count": len(history)},
                "E",
            )
            # #endregion
            return history
        except Exception as e:
            # #region debug instrumentation
            _debug_log(
                "conversation_store.py:103",
                "get_conversation_history exception",
                {"error": str(e), "error_type": type(e).__name__},
                "E",
            )
            # #endregion
            raise
        finally:
            if should_close:
                db.close()

    async def get_context_for_llm(self, session_id: str, limit: int = 5) -> str:
        """Get formatted conversation context for LLM."""
        # #region debug instrumentation
        _debug_log(
            "conversation_store.py:105",
            "get_context_for_llm entry",
            {"session_id": session_id, "limit": limit},
            "C",
        )
        # #endregion
        # FIXED: Now properly awaits the async method
        history = await self.get_conversation_history(session_id, limit=limit)
        # #region debug instrumentation
        _debug_log(
            "conversation_store.py:111",
            "get_conversation_history called (FIXED: now awaited)",
            {
                "history_type": type(history).__name__,
                "history_count": len(history) if history else 0,
            },
            "C",
        )
        # #endregion

        if not history:
            return ""

        context_parts = []
        for conv in history:
            context_parts.append(f"User: {conv['user_message']}")
            context_parts.append(f"Assistant: {conv['agent_response']}")

        return "\n".join(context_parts)


# Global conversation store
conversation_store = ConversationStore()
