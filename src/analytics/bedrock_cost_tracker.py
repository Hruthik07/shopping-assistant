"""Bedrock cost tracking via AWS Cost Explorer API."""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import time

from src.analytics.logger import logger
from src.utils.config import settings

# Bedrock model pricing (per 1M tokens)
# Format: "anthropic.claude-3-5-sonnet-20241022-v2:0"
BEDROCK_PRICING = {
    # Claude models on Bedrock
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input": 3.00,  # $3.00 per 1M input tokens
        "output": 15.00,  # $15.00 per 1M output tokens
        "provider": "bedrock",
    },
    "anthropic.claude-3-5-haiku-20241022-v2:0": {
        "input": 0.25,
        "output": 1.25,
        "provider": "bedrock",
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "input": 15.00,
        "output": 75.00,
        "provider": "bedrock",
    },
    # Add more Bedrock models as needed
}


class BedrockCostTracker:
    """Track Bedrock costs via AWS Cost Explorer API and token usage."""

    def __init__(self):
        """Initialize Bedrock cost tracker."""
        self.enabled = getattr(settings, "bedrock_enabled", False)
        self.region = getattr(settings, "aws_region", "us-east-1")
        self.client = None
        self.cost_history: List[Dict[str, Any]] = []
        self.daily_costs: Dict[str, float] = defaultdict(float)
        self.model_costs: Dict[str, float] = defaultdict(float)
        self.total_cost: float = 0.0

        if self.enabled:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize boto3 Cost Explorer client."""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError

            self.boto3 = boto3
            self.ClientError = ClientError
            self.NoCredentialsError = NoCredentialsError

            # Initialize Cost Explorer client
            self.client = boto3.client(
                "ce",  # Cost Explorer
                region_name="us-east-1",  # Cost Explorer is only available in us-east-1
            )
            logger.info("Bedrock cost tracker initialized (Cost Explorer)")
        except ImportError:
            logger.warning(
                "boto3 not installed. Bedrock cost tracking disabled. Install with: pip install boto3"
            )
            self.enabled = False
        except Exception as e:
            logger.warning(
                f"Failed to initialize Cost Explorer client: {e}. Bedrock cost tracking disabled."
            )
            self.enabled = False

    def calculate_cost_from_tokens(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Dict[str, Any]:
        """Calculate cost from token usage (if available from Bedrock response).

        Args:
            model: Bedrock model identifier (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost breakdown dictionary
        """
        # Normalize model name (remove version suffix if needed)
        pricing = BEDROCK_PRICING.get(model)
        if not pricing:
            # Try to find matching model without version
            for bedrock_model, bedrock_pricing in BEDROCK_PRICING.items():
                if bedrock_model.split(":")[0] in model:
                    pricing = bedrock_pricing
                    break

        if not pricing:
            # Fallback to default pricing
            pricing = {"input": 3.00, "output": 15.00, "provider": "bedrock"}
            logger.debug(f"Using default Bedrock pricing for model: {model}")

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
            "timestamp": time.time(),
        }

    async def get_bedrock_costs(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Query AWS Cost Explorer API for Bedrock costs.

        Args:
            start_date: Start date for cost query (defaults to 7 days ago)
            end_date: End date for cost query (defaults to today)

        Returns:
            Dictionary with cost breakdown
        """
        if not self.enabled or not self.client:
            return {"enabled": False, "error": "Cost Explorer not available"}

        try:
            # Default to last 7 days
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=7)

            # Query Cost Explorer
            response = self.client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
                Metrics=["UnblendedCost"],
                Filter={"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Bedrock"]}},
                GroupBy=[
                    {"Type": "DIMENSION", "Key": "SERVICE"},
                    {"Type": "DIMENSION", "Key": "USAGE_TYPE"},
                ],
            )

            # Parse response
            daily_costs = {}
            total_cost = 0.0

            for result in response.get("ResultsByTime", []):
                date = result["TimePeriod"]["Start"]
                cost = float(result["Total"]["UnblendedCost"]["Amount"])
                daily_costs[date] = cost
                total_cost += cost

            return {
                "enabled": True,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily_costs": daily_costs,
                "total_cost": round(total_cost, 4),
                "currency": response.get("ResultsByTime", [{}])[0]
                .get("Total", {})
                .get("UnblendedCost", {})
                .get("Unit", "USD"),
            }

        except self.NoCredentialsError:
            logger.warning("AWS credentials not found. Bedrock cost tracking disabled.")
            self.enabled = False
            return {"enabled": False, "error": "AWS credentials not found"}
        except self.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDenied":
                logger.warning("Cost Explorer access denied. Check IAM permissions.")
            else:
                logger.warning(f"Cost Explorer API error: {e}")
            return {"enabled": False, "error": f"Cost Explorer API error: {error_code}"}
        except Exception as e:
            logger.error(f"Unexpected error querying Cost Explorer: {e}")
            return {"enabled": False, "error": str(e)}

    def record_token_usage(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Dict[str, Any]:
        """Record token usage and calculate cost.

        Args:
            model: Bedrock model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost breakdown dictionary
        """
        cost_data = self.calculate_cost_from_tokens(model, input_tokens, output_tokens)

        # Update aggregates
        self.total_cost += cost_data["total_cost"]
        date = datetime.now().strftime("%Y-%m-%d")
        self.daily_costs[date] += cost_data["total_cost"]
        self.model_costs[model] += cost_data["total_cost"]

        # Add to history (keep last 1000 entries)
        self.cost_history.append(cost_data)
        if len(self.cost_history) > 1000:
            self.cost_history = self.cost_history[-1000:]

        return cost_data

    def get_cost_stats(self, days: int = 7, model: Optional[str] = None) -> Dict[str, Any]:
        """Get cost statistics.

        Args:
            days: Number of days to look back
            model: Optional model filter

        Returns:
            Cost statistics dictionary
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Filter by date and model
        recent_costs = [
            c
            for c in self.cost_history
            if c.get("date", "") >= cutoff_date and (not model or c["model"] == model)
        ]

        total_cost = sum(c["total_cost"] for c in recent_costs)
        total_input_tokens = sum(c["input_tokens"] for c in recent_costs)
        total_output_tokens = sum(c["output_tokens"] for c in recent_costs)

        # Model breakdown
        model_breakdown = defaultdict(float)
        for cost in recent_costs:
            model_breakdown[cost["model"]] += cost["total_cost"]

        return {
            "days": days,
            "total_cost": round(total_cost, 4),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_requests": len(recent_costs),
            "model_breakdown": dict(model_breakdown),
            "daily_average": round(total_cost / days, 4) if days > 0 else 0.0,
        }


# Global Bedrock cost tracker instance
bedrock_cost_tracker = BedrockCostTracker()
