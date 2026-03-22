"""Configuration management for the application."""

import json
import logging
from pydantic_settings import BaseSettings
from typing import Optional

_logger = logging.getLogger(__name__)

_DEFAULT_SECRET_KEY = "your-secret-key-change-in-production"


def _load_aws_secrets(secret_name: str, region: str) -> dict:
    """Load secrets from AWS Secrets Manager and return as a dict.

    Falls back to empty dict if boto3 is not installed or the secret cannot
    be retrieved (so local/dev environments continue to work without AWS).
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secret_string = response.get("SecretString", "{}")
        return json.loads(secret_string)
    except ImportError:
        _logger.warning("boto3 not installed – skipping AWS Secrets Manager load.")
        return {}
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Could not load AWS secret '%s': %s", secret_name, exc)
        return {}


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    openai_api_key: str = ""  # Made optional with default for graceful degradation
    anthropic_api_key: Optional[str] = None  # For Claude models
    tavily_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None  # For Google Shopping search
    etsy_api_key: Optional[str] = None  # For Etsy product data

    # Multi-source product APIs
    amazon_api_key: Optional[str] = None
    amazon_secret_key: Optional[str] = None
    amazon_associate_tag: Optional[str] = None
    ebay_api_key: Optional[str] = None
    walmart_api_key: Optional[str] = None
    bestbuy_api_key: Optional[str] = None
    pricegrabber_api_key: Optional[str] = None
    shopzilla_api_key: Optional[str] = None

    # Coupon/Promo APIs
    honey_api_key: Optional[str] = None
    retailmenot_api_key: Optional[str] = None
    couponfollow_api_key: Optional[str] = None

    # Exchange rate API (for currency conversion)
    exchangerate_api_key: Optional[str] = None  # exchangerate-api.com or similar

    # Database
    database_url: str = "sqlite:///./shopping_assistant.db"

    # LLM Configuration
    llm_provider: str = "anthropic"  # Options: "openai", "anthropic"
    llm_model: str = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku by default
    llm_temperature: float = 0.3  # Lower temperature for better tool calling

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 3565
    secret_key: str = _DEFAULT_SECRET_KEY
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # API Key Authentication (set this to protect endpoints)
    # Clients must send:  X-API-Key: <value>
    api_key: Optional[str] = None

    # Rate Limiting
    rate_limit_per_minute: int = 60

    # Redis Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True

    # Cache TTLs (in seconds)
    cache_llm_response_ttl: int = 3600  # 1 hour
    cache_product_search_ttl: int = 1800  # 30 minutes
    cache_session_ttl: int = 3600  # 1 hour
    cache_embedding_ttl: int = 86400  # 24 hours

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Langfuse Configuration
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_project_name: str = "shopping-assistant"
    langfuse_enabled: bool = True

    # DeepEval Configuration
    deepeval_api_key: Optional[str] = None
    deepeval_enabled: bool = True

    # Environment Configuration
    environment: str = "development"  # development, staging, production

    # Production Settings
    production_mode: bool = False  # Auto-detected from environment

    # Debug endpoints (disable in production unless explicitly enabled)
    debug_prompts: bool = False

    # CORS Configuration (for production)
    # In production set this to a comma-separated list of allowed origins.
    # Leaving it as "*" in production will cause a startup error.
    cors_origins: str = "*"

    # Model Routing Configuration
    enable_model_routing: bool = False  # Enable dynamic model selection (simple vs complex)

    # Context Configuration
    max_history_exchanges: int = 10  # Number of conversation exchanges to keep in context
    recent_exchanges_full: int = 5  # Recent exchanges with full context (500 chars)
    older_exchanges_truncate: int = 300  # Older exchanges truncate to N chars

    # Embedding Configuration (for semantic cache)
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    use_openai_embeddings: bool = False  # Use OpenAI embeddings instead of sentence transformers

    # Retrieval mode
    semantic_only_retrieval: bool = True

    # Product aggregation configuration
    product_source_priority: str = (
        "price_comparison,direct_retailers,serper"  # Comma-separated list
    )
    enable_price_comparison: bool = True
    enable_price_history: bool = True
    enable_coupon_integration: bool = True
    max_retailers_per_product: int = 5  # Maximum number of retailer options to show per product

    # AWS Configuration
    aws_region: str = "us-east-1"
    aws_secrets_name: Optional[str] = None   # e.g. "prod/shopping-assistant/secrets"
    cloudwatch_enabled: bool = False
    cloudwatch_namespace: str = "ShoppingAssistant/Application"
    bedrock_enabled: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Overlay secrets from AWS Secrets Manager when configured
        if self.aws_secrets_name:
            aws_secrets = _load_aws_secrets(self.aws_secrets_name, self.aws_region)
            for field_name, value in aws_secrets.items():
                lower_field = field_name.lower()
                if lower_field in self.model_fields and value is not None:
                    object.__setattr__(self, lower_field, value)

        # Auto-detect production mode
        self.production_mode = self.environment.lower() in ("production", "prod")

        # Adjust settings for production
        if self.production_mode:
            if self.log_level == "INFO":
                self.log_level = "WARNING"

            # Hard-fail if the default secret key is still in use
            if self.secret_key == _DEFAULT_SECRET_KEY:
                raise ValueError(
                    "SECRET_KEY is set to the default placeholder value. "
                    "Set a strong, random SECRET_KEY in your environment before "
                    "starting the application in production."
                )


settings = Settings()
