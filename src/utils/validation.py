"""Configuration and environment validation."""
import time
import json as json_module
from src.utils.config import settings
from src.analytics.logger import logger
from src.utils.debug_log import file_debug_log

# #region debug instrumentation
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    try:
        file_debug_log(location, message, data, hypothesis_id=hypothesis_id)
    except Exception:
        pass
# #endregion


def validate_config() -> dict:
    """Validate application configuration."""
    # #region debug instrumentation
    _debug_log('validation.py:6', 'validate_config called', {'llm_provider': settings.llm_provider}, 'M')
    # #endregion
    issues = []
    warnings = []
    
    # Critical validations - check based on LLM provider
    provider = settings.llm_provider.lower()
    # #region debug instrumentation
    _debug_log('validation.py:15', 'Checking LLM provider', {'provider': provider, 'has_anthropic_key': bool(settings.anthropic_api_key), 'has_openai_key': bool(settings.openai_api_key)}, 'M')
    # #endregion
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY is not set - agent will not function")
            # #region debug instrumentation
            _debug_log('validation.py:19', 'Missing ANTHROPIC_API_KEY', {}, 'M')
            # #endregion
    elif provider == "openai":
        if not settings.openai_api_key:
            issues.append("OPENAI_API_KEY is not set - agent will not function")
            # #region debug instrumentation
            _debug_log('validation.py:24', 'Missing OPENAI_API_KEY', {}, 'M')
            # #endregion
    else:
        issues.append(f"Invalid LLM provider: {provider}. Use 'anthropic' or 'openai'")
        # #region debug instrumentation
        _debug_log('validation.py:28', 'Invalid LLM provider', {'provider': provider}, 'M')
        # #endregion
    
    # Warning validations
    if not settings.serper_api_key:
        warnings.append("SERPER_API_KEY not set - Google Shopping API unavailable")
        # #region debug instrumentation
        _debug_log('validation.py:33', 'Missing SERPER_API_KEY', {}, 'M')
        # #endregion
    
    if not settings.tavily_api_key:
        warnings.append("TAVILY_API_KEY not set - web search unavailable")
        # #region debug instrumentation
        _debug_log('validation.py:37', 'Missing TAVILY_API_KEY', {}, 'M')
        # #endregion
    
    # Database validation
    if "sqlite" in settings.database_url:
        warnings.append("Using SQLite - not recommended for production")
        # #region debug instrumentation
        _debug_log('validation.py:42', 'Using SQLite database', {'database_url': settings.database_url}, 'M')
        # #endregion
    
    # Log results
    if issues:
        for issue in issues:
            logger.error(f"Configuration issue: {issue}")
    
    if warnings:
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    
    result = {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }
    # #region debug instrumentation
    _debug_log('validation.py:52', 'validate_config result', {'valid': result['valid'], 'issues_count': len(issues), 'warnings_count': len(warnings)}, 'M')
    # #endregion
    return result

