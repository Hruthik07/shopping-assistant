"""Optional file-based debug logging (disabled by default).

Some legacy instrumentation wrote directly to a hard-coded path like
`c:\\agentic_ai\\.cursor\\debug.log`, which is not appropriate for production.

Enable only when explicitly requested:
- FILE_DEBUG_LOG=true
- FILE_DEBUG_LOG_PATH=... (optional)
"""

import json
import os
import time
from typing import Any, Dict, Optional

from src.analytics.logger import logger


FILE_DEBUG_LOG_ENABLED = os.getenv("FILE_DEBUG_LOG", "false").lower() == "true"
FILE_DEBUG_LOG_PATH = os.getenv("FILE_DEBUG_LOG_PATH", r"c:\agentic_ai\.cursor\debug.log")


def file_debug_log(location: str, message: str, data: Dict[str, Any], hypothesis_id: Optional[str] = None):
    """Write a structured debug line to a file (opt-in)."""
    if not FILE_DEBUG_LOG_ENABLED:
        return
    try:
        payload = {
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
        }
        with open(FILE_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        # Never let debug logging break request handling
        logger.debug(f"[file_debug_log] failed: {e}")

