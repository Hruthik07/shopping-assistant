"""DeepEval configuration and setup."""
import os
from typing import Optional

# Conditional import - deepeval may not be compatible with all LangChain versions
try:
    from deepeval import evaluate
    from deepeval.models import GPTModel, DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEEPEVAL_AVAILABLE = False
    GPTModel = None
    DeepEvalBaseLLM = None
    evaluate = None

from src.utils.config import settings
from src.analytics.logger import logger


class DeepEvalConfig:
    """Configuration for DeepEval evaluation framework."""
    
    def __init__(self):
        """Initialize DeepEval configuration."""
        self.enabled = settings.deepeval_enabled and DEEPEVAL_AVAILABLE
        self.api_key = settings.deepeval_api_key or os.getenv("DEEPEVAL_API_KEY")
        self.evaluation_model: Optional[DeepEvalBaseLLM] = None
        
        if not DEEPEVAL_AVAILABLE:
            logger.warning("DeepEval is not available (import error). DeepEval features will be disabled.")
            self.enabled = False
            return
        
        if self.enabled:
            self._setup_evaluation_model()
    
    def _setup_evaluation_model(self):
        """Set up the evaluation model (judge model) for DeepEval."""
        if not DEEPEVAL_AVAILABLE or GPTModel is None:
            self.enabled = False
            return
            
        try:
            # Use GPT-4o-mini for cost-effective evaluation (20x cheaper than GPT-4!)
            # Only use GPT - no fallback to other models
            if not settings.openai_api_key:
                logger.error("OpenAI API key is required for DeepEval. Please set OPENAI_API_KEY in your .env file")
                self.enabled = False
                return
            
            # Using GPT-4o-mini for cost-effective evaluation (20x cheaper than GPT-4!)
            # Cost: ~$0.00015/1K input, $0.0006/1K output (vs GPT-4: $0.01-0.03/1K)
            # Alternative: "gpt-3.5-turbo" for even cheaper (~$0.0005/1K input)
            # Change to "gpt-4-turbo" if you need maximum accuracy (but 10x more expensive)
            self.evaluation_model = GPTModel(
                model="gpt-4o-mini",  # Cost-effective: 20x cheaper than GPT-4
                api_key=settings.openai_api_key
            )
            logger.info("DeepEval evaluation model configured: GPT-4o-mini (cost-effective, ~20x cheaper than GPT-4)")
            
        except Exception as e:
            logger.error(f"Failed to set up DeepEval evaluation model: {e}")
            self.enabled = False
    
    def get_evaluation_model(self) -> Optional[DeepEvalBaseLLM]:
        """Get the configured evaluation model."""
        return self.evaluation_model
    
    def is_enabled(self) -> bool:
        """Check if DeepEval is enabled."""
        return self.enabled


# Global DeepEval configuration
deepeval_config = DeepEvalConfig()

# Metric thresholds
METRIC_THRESHOLDS = {
    "faithfulness": 0.7,  # Similar to groundedness - response faithful to context
    "answer_relevancy": 0.7,
    "contextual_relevancy": 0.7,
    "contextual_precision": 0.7,
    "summarization": 0.7,
    "bias": 0.8,  # Higher threshold for bias (should be minimal)
    "hallucination": 0.7,  # Checks for made-up information
    "toxicity": 0.8  # Safety metric - should be minimal
}

