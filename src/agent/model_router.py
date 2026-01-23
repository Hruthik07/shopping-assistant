"""Model router for selecting appropriate LLM based on query complexity."""

from typing import Dict, Any, Optional, Tuple
import re

from src.analytics.logger import logger
from src.utils.config import settings


class ModelRouter:
    """Route queries to appropriate models based on complexity."""

    def __init__(self):
        self.simple_model = "claude-3-5-haiku-20241022"  # Cheaper, faster
        self.complex_model = "claude-3-5-sonnet-20241022"  # More capable, expensive
        self.openai_simple = "gpt-4o-mini"
        self.openai_complex = "gpt-4o"

        # Complexity indicators
        self.complex_indicators = [
            r"\b(compare|comparison|difference|versus|vs|better|best)\b",
            r"\b(explain|why|how|what is|describe|details)\b",
            r"\b(recommend|suggest|advice|opinion|preference)\b",
            r"\b(multiple|several|many|various|different)\b",
            r"\b(pros|cons|advantages|disadvantages|benefits)\b",
            r"\b(review|rating|quality|reliability|durability)\b",
            r"\b(technical|specification|specs|features|capabilities)\b",
            r"\b(which|should|would|could|might)\b",
        ]

        # Simple query patterns
        self.simple_indicators = [
            r"^find\s+me\s+",
            r"^show\s+me\s+",
            r"^i\s+need\s+",
            r"^i\s+want\s+",
            r"^search\s+for\s+",
            r"under\s+\$\d+",
            r"between\s+\$\d+\s+and\s+\$\d+",
        ]

        self.complexity_threshold = 2  # Number of complex indicators to classify as complex

    def classify_complexity(self, query: str, intent: Optional[Dict[str, Any]] = None) -> str:
        """Classify query as 'simple' or 'complex'.

        Args:
            query: User query
            intent: Optional intent detection result

        Returns:
            'simple' or 'complex'
        """
        query_lower = query.lower()

        # Check for simple patterns first
        simple_score = sum(
            1
            for pattern in self.simple_indicators
            if re.search(pattern, query_lower, re.IGNORECASE)
        )

        # Check for complex patterns
        complex_score = sum(
            1
            for pattern in self.complex_indicators
            if re.search(pattern, query_lower, re.IGNORECASE)
        )

        # Use intent if available
        if intent:
            intent_type = intent.get("type", "")
            if intent_type in ["comparison", "recommendation", "explanation"]:
                complex_score += 2
            elif intent_type == "simple_search":
                simple_score += 1

        # Check query length (longer queries tend to be more complex)
        if len(query.split()) > 15:
            complex_score += 1
        elif len(query.split()) < 5:
            simple_score += 1

        # Check for multiple tools needed (indicates complexity)
        if intent and intent.get("tools_needed"):
            tools_count = len(intent.get("tools_needed", []))
            if tools_count > 1:
                complex_score += 1

        # Classify based on scores
        if complex_score >= self.complexity_threshold:
            return "complex"
        elif simple_score > 0 and complex_score == 0:
            return "simple"
        else:
            # Default to simple for cost savings (can be adjusted)
            return "simple"

    def select_model(
        self, query: str, intent: Optional[Dict[str, Any]] = None, force_model: Optional[str] = None
    ) -> Tuple[str, str]:
        """Select appropriate model for the query.

        Args:
            query: User query
            intent: Optional intent detection result
            force_model: Optional model to force (overrides routing)

        Returns:
            Tuple of (model_name, complexity_level)
        """
        # If model is forced, use it
        if force_model:
            return force_model, "forced"

        # Check if model routing is enabled
        if not getattr(settings, "enable_model_routing", True):
            return settings.llm_model, "default"

        # Classify complexity
        complexity = self.classify_complexity(query, intent)

        # Select model based on provider and complexity
        provider = settings.llm_provider.lower()

        if provider == "anthropic":
            if complexity == "simple":
                selected_model = self.simple_model
            else:
                selected_model = self.complex_model
        elif provider == "openai":
            if complexity == "simple":
                selected_model = self.openai_simple
            else:
                selected_model = self.openai_complex
        else:
            # Fallback to configured model
            selected_model = settings.llm_model
            complexity = "default"

        logger.debug(
            f"Model routing: query='{query[:50]}...' -> complexity={complexity} -> model={selected_model}"
        )

        return selected_model, complexity

    def get_cost_savings_estimate(
        self, simple_queries: int, complex_queries: int
    ) -> Dict[str, Any]:
        """Estimate cost savings from model routing.

        Args:
            simple_queries: Number of simple queries
            complex_queries: Number of complex queries

        Returns:
            Dictionary with cost savings breakdown
        """
        # Average tokens per query (rough estimate)
        avg_input_tokens = 1000
        avg_output_tokens = 500

        provider = settings.llm_provider.lower()

        if provider == "anthropic":
            # Haiku pricing
            simple_input_cost = (avg_input_tokens / 1_000_000) * 0.25
            simple_output_cost = (avg_output_tokens / 1_000_000) * 1.25
            simple_total = simple_input_cost + simple_output_cost

            # Sonnet pricing
            complex_input_cost = (avg_input_tokens / 1_000_000) * 3.00
            complex_output_cost = (avg_output_tokens / 1_000_000) * 15.00
            complex_total = complex_input_cost + complex_output_cost

            # Cost if all used Sonnet
            all_sonnet_cost = (simple_queries + complex_queries) * complex_total

            # Cost with routing
            routed_cost = (simple_queries * simple_total) + (complex_queries * complex_total)

            savings = all_sonnet_cost - routed_cost
            savings_percent = (savings / all_sonnet_cost * 100) if all_sonnet_cost > 0 else 0

        elif provider == "openai":
            # GPT-4o-mini pricing
            simple_input_cost = (avg_input_tokens / 1_000_000) * 0.15
            simple_output_cost = (avg_output_tokens / 1_000_000) * 0.60
            simple_total = simple_input_cost + simple_output_cost

            # GPT-4o pricing
            complex_input_cost = (avg_input_tokens / 1_000_000) * 2.50
            complex_output_cost = (avg_output_tokens / 1_000_000) * 10.00
            complex_total = complex_input_cost + complex_output_cost

            # Cost if all used GPT-4o
            all_complex_cost = (simple_queries + complex_queries) * complex_total

            # Cost with routing
            routed_cost = (simple_queries * simple_total) + (complex_queries * complex_total)

            savings = all_complex_cost - routed_cost
            savings_percent = (savings / all_complex_cost * 100) if all_complex_cost > 0 else 0
        else:
            return {
                "error": "Cost savings calculation not available for this provider",
                "provider": provider,
            }

        return {
            "simple_queries": simple_queries,
            "complex_queries": complex_queries,
            "total_queries": simple_queries + complex_queries,
            "estimated_savings": round(savings, 4),
            "savings_percent": round(savings_percent, 2),
            "routed_cost": round(routed_cost, 4),
            "all_complex_cost": round(
                all_sonnet_cost if provider == "anthropic" else all_complex_cost, 4
            ),
        }


# Global model router instance
model_router = ModelRouter()
