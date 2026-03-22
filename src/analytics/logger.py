"""Structured logging for the application.

Output format is controlled by the LOG_FORMAT environment variable:
  LOG_FORMAT=json  (default) – one JSON object per line, ideal for CloudWatch Logs Insights
  LOG_FORMAT=text            – human-readable plain text for local development
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from src.utils.config import settings


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    CloudWatch Logs Insights can query structured fields directly, e.g.:
        fields @timestamp, level, message | filter level = "ERROR"
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text

        try:
            return json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fall back if any value is not JSON-serialisable
            payload["message"] = repr(record.getMessage())
            return json.dumps(payload, ensure_ascii=False)


def _build_formatter(use_json: bool, include_location: bool = False) -> logging.Formatter:
    if use_json:
        return JsonFormatter()
    fmt = (
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        if include_location
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.Formatter(fmt)


def setup_logger(
    name: str = "shopping_assistant",
    log_file: Optional[str] = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """Set up structured logger.

    Uses JSON format by default (overridable via LOG_FORMAT=text).
    """
    use_json = os.getenv("LOG_FORMAT", "json").strip().lower() != "text"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    # Console handler – always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_build_formatter(use_json, include_location=False))
    logger.addHandler(console_handler)

    # File handler – optional, includes function/line context
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_build_formatter(use_json, include_location=True))
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger(log_file=settings.log_file, log_level=settings.log_level)
