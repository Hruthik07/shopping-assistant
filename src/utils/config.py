"""Configuration management for the application."""
from pydantic_settings import BaseSettings
from typing import Optional


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
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Redis Cache
    redis_url: str = "redis://localhost:8971/0"
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
    cors_origins: str = "*"  # Comma-separated list of allowed origins
    
    # Model Routing Configuration
    enable_model_routing: bool = False  # Enable dynamic model selection (simple vs complex)
    
    # Context Configuration
    max_history_exchanges: int = 10  # Number of conversation exchanges to keep in context
    recent_exchanges_full: int = 5   # Recent exchanges with full context (500 chars)
    older_exchanges_truncate: int = 300  # Older exchanges truncate to N chars
    
    # Embedding Configuration (for semantic cache)
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    use_openai_embeddings: bool = False  # Use OpenAI embeddings instead of sentence transformers

    # Retrieval mode
    # If enabled, prefer semantic mechanisms over keyword/term-based shortcuts:
    # - Skip exact-match caches (use semantic cache only)
    # - Disable brand/keyword heuristic filtering
    # - Prefer semantic similarity scores for product ranking/explanations
    semantic_only_retrieval: bool = True
    
    # Product aggregation configuration
    # Data source priority (customer-first order: price comparison first, then direct retailers, then serper)
    product_source_priority: str = "price_comparison,direct_retailers,serper"  # Comma-separated list
    enable_price_comparison: bool = True
    enable_price_history: bool = True
    enable_coupon_integration: bool = True
    max_retailers_per_product: int = 5  # Maximum number of retailer options to show per product
    
    # AWS Configuration (for Bedrock deployment)
    aws_region: str = "us-east-1"  # AWS region for Bedrock/CloudWatch
    cloudwatch_enabled: bool = False  # Enable CloudWatch metrics export
    cloudwatch_namespace: str = "ShoppingAssistant/Application"  # CloudWatch namespace
    bedrock_enabled: bool = False  # Enable AWS Bedrock as LLM provider
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect production mode
        self.production_mode = (
            self.environment.lower() == "production" or
            self.environment.lower() == "prod"
        )
        
        # Adjust settings for production
        if self.production_mode:
            # More restrictive logging in production
            if self.log_level == "INFO":
                self.log_level = "WARNING"
            
            # Require secret key change
            if self.secret_key == "your-secret-key-change-in-production":
                import warnings
                warnings.warn(
                    "WARNING: Using default secret key in production! "
                    "Change SECRET_KEY in .env file immediately."
                )


settings = Settings()

