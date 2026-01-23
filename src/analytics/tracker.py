"""Agent decision tracking and analytics."""

from typing import Dict, Any, List
from datetime import datetime
from src.analytics.logger import logger


class AgentTracker:
    """Track agent decisions and performance."""

    def __init__(self):
        self.decisions: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "total_queries": 0,
            "tool_calls": 0,
            "average_response_time": 0.0,
            # Deal detection metrics
            "deals_detected": 0,
            "total_products_analyzed": 0,
            "deal_detection_rate": 0.0,
            "average_savings_percent": 0.0,
            "total_savings_amount": 0.0,
            # Price comparison metrics
            "price_comparisons": 0,
            "products_with_multiple_retailers": 0,
            "average_retailers_per_product": 0.0,
            "average_price_difference": 0.0,
            # API call metrics
            "api_calls_total": 0,
            "api_calls_successful": 0,
            "api_calls_failed": 0,
            "api_success_rate": 0.0,
        }

    def track_decision(self, query: str, tool_used: str, result: Any, response_time: float):
        """Track an agent decision."""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "tool_used": tool_used,
            "result_summary": str(result)[:200] if result else None,
            "response_time": response_time,
        }
        self.decisions.append(decision)

        # Update metrics
        self.metrics["total_queries"] += 1
        if tool_used:
            self.metrics["tool_calls"] += 1

        # Update average response time
        total_time = self.metrics["average_response_time"] * (self.metrics["total_queries"] - 1)
        self.metrics["average_response_time"] = (total_time + response_time) / self.metrics[
            "total_queries"
        ]

        logger.info(f"Agent decision tracked: {tool_used} for query: {query[:50]}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent agent decisions."""
        return self.decisions[-limit:]

    def track_deal_detection(
        self,
        products_analyzed: int,
        deals_found: int,
        total_savings: float,
        average_savings_percent: float,
    ):
        """Track deal detection metrics."""
        self.metrics["total_products_analyzed"] += products_analyzed
        self.metrics["deals_detected"] += deals_found
        self.metrics["total_savings_amount"] += total_savings

        # Update deal detection rate
        if self.metrics["total_products_analyzed"] > 0:
            self.metrics["deal_detection_rate"] = (
                self.metrics["deals_detected"] / self.metrics["total_products_analyzed"]
            ) * 100

        # Update average savings (weighted average)
        if deals_found > 0:
            current_avg = self.metrics["average_savings_percent"]
            current_count = self.metrics["deals_detected"] - deals_found
            if current_count > 0:
                total_avg = (
                    current_avg * current_count + average_savings_percent * deals_found
                ) / self.metrics["deals_detected"]
            else:
                total_avg = average_savings_percent
            self.metrics["average_savings_percent"] = total_avg

        logger.debug(f"Deal detection tracked: {deals_found}/{products_analyzed} deals found")

    def track_price_comparison(
        self,
        products_compared: int,
        products_with_multiple_retailers: int,
        average_retailers: float,
        average_price_diff: float,
    ):
        """Track price comparison metrics."""
        self.metrics["price_comparisons"] += products_compared
        self.metrics["products_with_multiple_retailers"] += products_with_multiple_retailers

        # Update average retailers per product
        if self.metrics["price_comparisons"] > 0:
            current_avg = self.metrics["average_retailers_per_product"]
            current_count = self.metrics["price_comparisons"] - products_compared
            if current_count > 0:
                total_avg = (
                    current_avg * current_count + average_retailers * products_compared
                ) / self.metrics["price_comparisons"]
            else:
                total_avg = average_retailers
            self.metrics["average_retailers_per_product"] = total_avg

        # Update average price difference
        if products_compared > 0:
            current_avg = self.metrics["average_price_difference"]
            current_count = self.metrics["price_comparisons"] - products_compared
            if current_count > 0:
                total_avg = (
                    current_avg * current_count + average_price_diff * products_compared
                ) / self.metrics["price_comparisons"]
            else:
                total_avg = average_price_diff
            self.metrics["average_price_difference"] = total_avg

        logger.debug(
            f"Price comparison tracked: {products_compared} products, {products_with_multiple_retailers} with multiple retailers"
        )

    def track_api_call(self, success: bool):
        """Track API call success/failure."""
        self.metrics["api_calls_total"] += 1
        if success:
            self.metrics["api_calls_successful"] += 1
        else:
            self.metrics["api_calls_failed"] += 1

        # Update success rate
        if self.metrics["api_calls_total"] > 0:
            self.metrics["api_success_rate"] = (
                self.metrics["api_calls_successful"] / self.metrics["api_calls_total"]
            ) * 100


# Global tracker instance
tracker = AgentTracker()
