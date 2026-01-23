"""Utility helper functions."""

import time
import json as json_module
from typing import Any, Dict, List
import json
from datetime import datetime
from src.utils.debug_log import file_debug_log


# #region debug instrumentation
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    try:
        file_debug_log(location, message, data, hypothesis_id=hypothesis_id)
    except Exception:
        pass


# #endregion


def format_product_response(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format product data for API response."""
    formatted = []
    for product in products:
        formatted.append(
            {
                "id": product.get("id"),
                "name": product.get("name"),
                "description": product.get("description"),
                "price": product.get("price"),
                "category": product.get("category"),
                "rating": product.get("rating"),
                "image_url": product.get("image_url"),
                "availability": product.get("availability", True),
            }
        )
    return formatted


def calculate_similarity_score(query: str, text: str) -> float:
    """Calculate simple text similarity score."""
    query_lower = query.lower()
    text_lower = text.lower()

    # Simple word overlap
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())

    if not query_words:
        return 0.0

    intersection = query_words.intersection(text_words)
    return len(intersection) / len(query_words)


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON string."""
    # #region debug instrumentation
    _debug_log(
        "helpers.py:47",
        "safe_json_loads called",
        {"text_length": len(text) if text else 0, "text_preview": text[:50] if text else None},
        "L",
    )
    # #endregion
    try:
        result = json.loads(text)
        # #region debug instrumentation
        _debug_log(
            "helpers.py:51", "safe_json_loads success", {"result_type": type(result).__name__}, "L"
        )
        # #endregion
        return result
    except (json.JSONDecodeError, TypeError) as e:
        # #region debug instrumentation
        _debug_log(
            "helpers.py:55",
            "safe_json_loads failed",
            {"error": str(e), "error_type": type(e).__name__, "using_default": default is not None},
            "L",
        )
        # #endregion
        return default


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().isoformat()
