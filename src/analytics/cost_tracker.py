"""Cost tracking and calculation for LLM usage."""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import time

from src.analytics.logger import logger
from src.utils.config import settings


# Pricing per 1M tokens (as of 2024)
PRICING = {
    # Anthropic Claude models
    "claude-3-5-haiku-20241022": {
        "input": 0.25,  # $0.25 per 1M input tokens
        "output": 1.25,  # $1.25 per 1M output tokens
        "provider": "anthropic"
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
        "provider": "anthropic"
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
        "provider": "anthropic"
    },
    # OpenAI models
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
        "provider": "openai"
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "provider": "openai"
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
        "provider": "openai"
    },
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50,
        "provider": "openai"
    },
    # AWS Bedrock models (via Bedrock API)
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input": 3.00,
        "output": 15.00,
        "provider": "bedrock"
    },
    "anthropic.claude-3-5-haiku-20241022-v2:0": {
        "input": 0.25,
        "output": 1.25,
        "provider": "bedrock"
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "input": 15.00,
        "output": 75.00,
        "provider": "bedrock"
    },
    # Default fallback (use cheapest)
    "default": {
        "input": 0.25,
        "output": 1.25,
        "provider": "anthropic"
    }
}


class CostTracker:
    """Track LLM costs based on token usage."""
    
    def __init__(self):
        self.cost_history: List[Dict[str, Any]] = []
        self.daily_costs: Dict[str, float] = defaultdict(float)  # date -> total cost
        self.model_costs: Dict[str, float] = defaultdict(float)  # model -> total cost
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.request_count: int = 0
        
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, Any]:
        """Calculate cost for a single LLM call.
        
        Args:
            model: Model name (e.g., "claude-3-5-haiku-20241022")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with cost breakdown
        """
        # Get pricing for model, fallback to default
        pricing = PRICING.get(model, PRICING["default"])
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "model": model,
            "provider": pricing["provider"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "timestamp": time.time()
        }
    
    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record token usage and calculate cost.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_id: Optional request ID for tracking
            query: Optional query text for context
            
        Returns:
            Cost breakdown dictionary
        """
        cost_data = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Add metadata
        cost_data["request_id"] = request_id
        cost_data["query"] = query[:100] if query else None  # Truncate for storage
        cost_data["date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Update aggregates
        self.total_cost += cost_data["total_cost"]
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1
        
        # Update daily costs
        self.daily_costs[cost_data["date"]] += cost_data["total_cost"]
        
        # Update model costs
        self.model_costs[model] += cost_data["total_cost"]
        
        # Add to history (keep last 1000 entries)
        self.cost_history.append(cost_data)
        if len(self.cost_history) > 1000:
            self.cost_history = self.cost_history[-1000:]
        
        logger.debug(
            f"Cost recorded: ${cost_data['total_cost']:.6f} for {model} "
            f"({input_tokens} input, {output_tokens} output tokens)"
        )
        
        return cost_data
    
    def get_cost_stats(
        self,
        days: int = 7,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost statistics for a time period.
        
        Args:
            days: Number of days to look back
            model: Optional model filter
            
        Returns:
            Dictionary with cost statistics
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Filter history
        filtered_history = [
            entry for entry in self.cost_history
            if entry["date"] >= cutoff_date
        ]
        
        if model:
            filtered_history = [
                entry for entry in filtered_history
                if entry["model"] == model
            ]
        
        # Calculate statistics
        total_cost = sum(entry["total_cost"] for entry in filtered_history)
        total_input_tokens = sum(entry["input_tokens"] for entry in filtered_history)
        total_output_tokens = sum(entry["output_tokens"] for entry in filtered_history)
        total_tokens = total_input_tokens + total_output_tokens
        request_count = len(filtered_history)
        
        # Daily breakdown
        daily_breakdown = defaultdict(float)
        for entry in filtered_history:
            daily_breakdown[entry["date"]] += entry["total_cost"]
        
        # Model breakdown
        model_breakdown = defaultdict(float)
        for entry in filtered_history:
            model_breakdown[entry["model"]] += entry["total_cost"]
        
        return {
            "period_days": days,
            "total_cost": round(total_cost, 4),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "request_count": request_count,
            "average_cost_per_request": round(total_cost / request_count, 6) if request_count > 0 else 0,
            "average_tokens_per_request": round(total_tokens / request_count, 2) if request_count > 0 else 0,
            "daily_breakdown": dict(sorted(daily_breakdown.items())),
            "model_breakdown": dict(model_breakdown),
            "all_time_total": round(self.total_cost, 4),
            "all_time_requests": self.request_count
        }
    
    def get_recent_costs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent cost records.
        
        Args:
            limit: Number of recent records to return
            
        Returns:
            List of recent cost records
        """
        return sorted(
            self.cost_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def check_budget_alert(
        self,
        daily_budget: Optional[float] = None,
        weekly_budget: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Check if cost exceeds budget thresholds.
        
        Args:
            daily_budget: Daily budget limit in USD
            weekly_budget: Weekly budget limit in USD
            
        Returns:
            Alert dictionary if threshold exceeded, None otherwise
        """
        alerts = []
        
        if daily_budget:
            today = datetime.now().strftime("%Y-%m-%d")
            today_cost = self.daily_costs.get(today, 0.0)
            if today_cost >= daily_budget:
                alerts.append({
                    "type": "daily_budget_exceeded",
                    "threshold": daily_budget,
                    "actual": round(today_cost, 4),
                    "date": today
                })
        
        if weekly_budget:
            weekly_stats = self.get_cost_stats(days=7)
            weekly_cost = weekly_stats["total_cost"]
            if weekly_cost >= weekly_budget:
                alerts.append({
                    "type": "weekly_budget_exceeded",
                    "threshold": weekly_budget,
                    "actual": round(weekly_cost, 4),
                    "period": "7 days"
                })
        
        return alerts[0] if alerts else None


# Global cost tracker instance
cost_tracker = CostTracker()
