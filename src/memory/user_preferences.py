"""User preference tracking and learning."""

import time
import json as json_module
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from src.database.models import UserPreference, User
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


class UserPreferenceTracker:
    """Track and learn user preferences."""

    def get_preferences(self, user_id: int, db: Optional[Session] = None) -> Dict[str, Any]:
        """Get user preferences."""
        # #region debug instrumentation
        _debug_log(
            "user_preferences.py:12",
            "get_preferences entry",
            {"user_id": user_id, "has_db": db is not None},
            "D",
        )
        # #endregion
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            preferences = db.query(UserPreference).filter(UserPreference.user_id == user_id).all()
            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:24",
                "Preferences queried",
                {"preferences_count": len(preferences)},
                "D",
            )
            # #endregion

            if not preferences:
                # #region debug instrumentation
                _debug_log(
                    "user_preferences.py:28",
                    "No preferences found - returning empty",
                    {"user_id": user_id},
                    "D",
                )
                # #endregion
                return {}

            # Combine preferences
            combined = {
                "categories": [],
                "brands": [],
                "price_range": {"min": None, "max": None},
                "other": {},
            }

            for pref in preferences:
                if pref.category:
                    combined["categories"].append(pref.category)
                if pref.brand:
                    combined["brands"].append(pref.brand)
                if pref.price_range_min:
                    combined["price_range"]["min"] = pref.price_range_min
                if pref.price_range_max:
                    combined["price_range"]["max"] = pref.price_range_max
                if pref.preferences:
                    combined["other"].update(pref.preferences)

            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:51",
                "get_preferences returning",
                {
                    "combined_keys": list(combined.keys()),
                    "categories_count": len(combined["categories"]),
                    "brands_count": len(combined["brands"]),
                },
                "D",
            )
            # #endregion
            return combined
        except Exception as e:
            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:54",
                "get_preferences exception",
                {"error": str(e), "error_type": type(e).__name__},
                "D",
            )
            # #endregion
            raise
        finally:
            if should_close:
                db.close()

    def update_preferences(
        self,
        user_id: int,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        price_range_min: Optional[float] = None,
        price_range_max: Optional[float] = None,
        other_preferences: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None,
    ):
        """Update user preferences."""
        # #region debug instrumentation
        _debug_log(
            "user_preferences.py:56",
            "update_preferences entry",
            {
                "user_id": user_id,
                "category": category,
                "brand": brand,
                "price_range_min": price_range_min,
                "price_range_max": price_range_max,
            },
            "D",
        )
        # #endregion
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            # Check if preference exists
            # BUG: Filter condition is wrong - "if category else True" will always match when category is None
            filter_condition = UserPreference.category == category if category else True
            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:74",
                "Filter condition",
                {"category": category, "filter_condition_type": type(filter_condition).__name__},
                "D",
            )
            # #endregion
            existing = (
                db.query(UserPreference)
                .filter(UserPreference.user_id == user_id, filter_condition)
                .first()
            )
            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:78",
                "Existing preference query",
                {"existing_found": existing is not None},
                "D",
            )
            # #endregion

            if existing:
                if brand:
                    existing.brand = brand
                if price_range_min:
                    existing.price_range_min = price_range_min
                if price_range_max:
                    existing.price_range_max = price_range_max
                if other_preferences:
                    if existing.preferences:
                        existing.preferences.update(other_preferences)
                    else:
                        existing.preferences = other_preferences
            else:
                preference = UserPreference(
                    user_id=user_id,
                    category=category,
                    brand=brand,
                    price_range_min=price_range_min,
                    price_range_max=price_range_max,
                    preferences=other_preferences,
                )
                db.add(preference)

            db.commit()
            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:102",
                "Preferences updated successfully",
                {"user_id": user_id},
                "D",
            )
            # #endregion
            logger.info(f"Updated preferences for user: {user_id}")
        except Exception as e:
            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:105",
                "update_preferences exception",
                {"error": str(e), "error_type": type(e).__name__},
                "D",
            )
            # #endregion
            raise
        finally:
            if should_close:
                db.close()

    def learn_from_interaction(
        self, user_id: Optional[int], query: str, selected_products: List[Dict[str, Any]]
    ):
        """Learn preferences from user interactions."""
        # #region debug instrumentation
        _debug_log(
            "user_preferences.py:108",
            "learn_from_interaction entry",
            {
                "user_id": user_id,
                "query": query,
                "products_count": len(selected_products) if selected_products else 0,
            },
            "D",
        )
        # #endregion
        if not user_id or not selected_products:
            # #region debug instrumentation
            _debug_log(
                "user_preferences.py:115",
                "learn_from_interaction early return",
                {"has_user_id": user_id is not None, "has_products": bool(selected_products)},
                "D",
            )
            # #endregion
            return

        # Extract preferences from selected products
        categories = set()
        brands = set()
        prices = []

        for product in selected_products:
            if "category" in product:
                categories.add(product["category"])
            if "brand" in product:
                brands.add(product["brand"])
            if "price" in product:
                price = product["price"]
                # #region debug instrumentation
                _debug_log(
                    "user_preferences.py:128",
                    "Price extraction",
                    {"price": price, "price_type": type(price).__name__},
                    "D",
                )
                # #endregion
                # Handle string prices like "$29.99"
                if isinstance(price, str):
                    try:
                        import re

                        price_match = re.search(r"(\d+(?:\.\d+)?)", str(price))
                        if price_match:
                            price = float(price_match.group(1))
                        else:
                            continue
                    except:
                        continue
                prices.append(price)

        # Update preferences
        if categories:
            for category in categories:
                self.update_preferences(user_id=user_id, category=category)

        if prices:
            avg_price = sum(prices) / len(prices)
            self.update_preferences(
                user_id=user_id, price_range_min=avg_price * 0.7, price_range_max=avg_price * 1.3
            )


# Global preference tracker
preference_tracker = UserPreferenceTracker()
