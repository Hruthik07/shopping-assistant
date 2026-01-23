"""Session management for user conversations."""

import uuid
import time
import json as json_module
from typing import Optional
from sqlalchemy.orm import Session
from src.database.models import Session as SessionModel
from src.database.db import SessionLocal
from src.analytics.logger import logger
from src.utils.debug_log import file_debug_log


# #region debug instrumentation
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    try:
        file_debug_log(location, message, data, hypothesis_id=hypothesis_id)
    except Exception:
        pass


# #endregion


class SessionManager:
    """Manage user sessions."""

    def create_session(self, user_id: Optional[int] = None, db: Optional[Session] = None) -> str:
        """Create a new session."""
        # #region debug instrumentation
        _debug_log(
            "session_manager.py:13",
            "create_session entry",
            {"user_id": user_id, "has_db": db is not None},
            "B",
        )
        # #endregion
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            session_id = str(uuid.uuid4())
            session = SessionModel(session_id=session_id, user_id=user_id)
            db.add(session)
            db.commit()
            db.refresh(session)
            # #region debug instrumentation
            _debug_log(
                "session_manager.py:28",
                "Session created successfully",
                {
                    "session_id": session_id,
                    "session_db_id": session.id if hasattr(session, "id") else "N/A",
                },
                "B",
            )
            # #endregion

            logger.info(f"Created new session: {session_id}")
            return session_id
        except Exception as e:
            db.rollback()
            # #region debug instrumentation
            _debug_log(
                "session_manager.py:33",
                "create_session exception",
                {"error": str(e), "error_type": type(e).__name__},
                "B",
            )
            # #endregion
            logger.error(f"Error creating session: {e}")
            raise
        finally:
            if should_close:
                db.close()

    def get_session(self, session_id: str, db: Optional[Session] = None) -> Optional[SessionModel]:
        """Get session by ID."""
        # #region debug instrumentation
        _debug_log(
            "session_manager.py:40",
            "get_session entry",
            {"session_id": session_id, "has_db": db is not None},
            "B",
        )
        # #endregion
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            # #region debug instrumentation
            _debug_log(
                "session_manager.py:48",
                "get_session result",
                {"session_found": session is not None, "session_id": session_id},
                "B",
            )
            # #endregion
            return session
        except Exception as e:
            # #region debug instrumentation
            _debug_log(
                "session_manager.py:51",
                "get_session exception",
                {"error": str(e), "error_type": type(e).__name__},
                "B",
            )
            # #endregion
            raise
        finally:
            if should_close:
                db.close()

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> str:
        """Get existing session or create new one."""
        # #region debug instrumentation
        _debug_log(
            "session_manager.py:56",
            "get_or_create_session entry",
            {"session_id": session_id, "user_id": user_id, "has_db": db is not None},
            "B",
        )
        # #endregion
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            if session_id:
                session = self.get_session(session_id, db)
                if session:
                    # #region debug instrumentation
                    _debug_log(
                        "session_manager.py:61",
                        "Existing session found",
                        {"session_id": session_id},
                        "B",
                    )
                    # #endregion
                    return session_id

                # Session doesn't exist, create it with the provided session_id
                logger.info(f"Creating new session with provided ID: {session_id}")
                # #region debug instrumentation
                _debug_log(
                    "session_manager.py:67",
                    "Creating session with provided ID",
                    {"session_id": session_id, "user_id": user_id},
                    "B",
                )
                # #endregion
                new_session = SessionModel(session_id=session_id, user_id=user_id)
                db.add(new_session)
                db.commit()
                db.refresh(new_session)
                logger.info(f"Created session: {session_id}")
                return session_id

            # No session_id provided, create new one
            # #region debug instrumentation
            _debug_log("session_manager.py:67", "Creating new session", {"user_id": user_id}, "B")
            # #endregion
            return self.create_session(user_id, db=db)
        except Exception as e:
            db.rollback()
            logger.error(f"Error in get_or_create_session: {e}")
            # Fallback to creating new session
            if session_id:
                return self.create_session(user_id)
            raise
        finally:
            if should_close:
                db.close()


# Global session manager
session_manager = SessionManager()
