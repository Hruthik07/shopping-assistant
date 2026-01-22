"""Retry utilities with exponential backoff and circuit breaker support."""
import asyncio
import time
import json as json_module
from typing import Callable, Any, Optional, TypeVar, List
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
    RetryCallState
)
from circuitbreaker import circuit
import httpx
from src.analytics.logger import logger
from src.utils.debug_log import file_debug_log

# #region debug instrumentation
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    try:
        file_debug_log(location, message, data, hypothesis_id=hypothesis_id)
    except Exception:
        pass
# #endregion

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 10.0,
        exponential_base: float = 2.0,
        retry_on: Optional[List[Exception]] = None
    ):
        self.max_attempts = max_attempts
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        self.exponential_base = exponential_base
        self.retry_on = retry_on or [
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
            ConnectionError,
            TimeoutError
        ]


def should_retry_http_error(exception: Exception) -> bool:
    """Determine if HTTP error should be retried."""
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        # Retry on 5xx errors and 429 (rate limit)
        return status_code >= 500 or status_code == 429
    return True


def async_retry(
    config: Optional[RetryConfig] = None,
    fallback: Optional[Callable] = None
):
    """
    Decorator for async functions with retry logic.
    
    Args:
        config: Retry configuration
        fallback: Fallback function to call if all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # #region debug instrumentation
            _debug_log('retry.py:67', 'Async retry wrapper called', {'func_name': func.__name__, 'max_attempts': config.max_attempts, 'has_fallback': fallback is not None}, 'J')
            # #endregion
            retry_decorator = retry(
                stop=stop_after_attempt(config.max_attempts),
                wait=wait_exponential(
                    multiplier=config.initial_wait,
                    max=config.max_wait,
                    exp_base=config.exponential_base
                ),
                retry=retry_if_exception_type(tuple(config.retry_on)),
                reraise=True,
                before_sleep=lambda retry_state: (
                    logger.warning(
                        f"Retrying {func.__name__} after {retry_state.outcome.exception()}"
                        f" (attempt {retry_state.attempt_number}/{config.max_attempts})"
                    ),
                    _debug_log('retry.py:79', 'Retry attempt', {'func_name': func.__name__, 'attempt': retry_state.attempt_number, 'max_attempts': config.max_attempts, 'error': str(retry_state.outcome.exception())}, 'J')
                )[1] if retry_state.outcome.exception() else None
            )
            
            try:
                result = await retry_decorator(func)(*args, **kwargs)
                # #region debug instrumentation
                _debug_log('retry.py:86', 'Async retry succeeded', {'func_name': func.__name__, 'result_type': type(result).__name__}, 'J')
                # #endregion
                return result
            except Exception as e:
                logger.error(f"All retries exhausted for {func.__name__}: {e}")
                # #region debug instrumentation
                _debug_log('retry.py:90', 'All retries exhausted', {'func_name': func.__name__, 'error': str(e), 'error_type': type(e).__name__, 'has_fallback': fallback is not None}, 'J')
                # #endregion
                if fallback:
                    logger.info(f"Attempting fallback for {func.__name__}")
                    try:
                        fallback_result = await fallback(*args, **kwargs)
                        # #region debug instrumentation
                        _debug_log('retry.py:96', 'Fallback succeeded', {'func_name': func.__name__}, 'J')
                        # #endregion
                        return fallback_result
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        # #region debug instrumentation
                        _debug_log('retry.py:101', 'Fallback failed', {'func_name': func.__name__, 'fallback_error': str(fallback_error)}, 'J')
                        # #endregion
                        raise e  # Raise original error
                raise
        
        return wrapper
    return decorator


def sync_retry(
    config: Optional[RetryConfig] = None,
    fallback: Optional[Callable] = None
):
    """
    Decorator for sync functions with retry logic.
    
    Args:
        config: Retry configuration
        fallback: Fallback function to call if all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_decorator = retry(
                stop=stop_after_attempt(config.max_attempts),
                wait=wait_exponential(
                    multiplier=config.initial_wait,
                    max=config.max_wait,
                    exp_base=config.exponential_base
                ),
                retry=retry_if_exception_type(tuple(config.retry_on)),
                reraise=True,
                before_sleep=lambda retry_state: logger.warning(
                    f"Retrying {func.__name__} after {retry_state.outcome.exception()}"
                    f" (attempt {retry_state.attempt_number}/{config.max_attempts})"
                )
            )
            
            try:
                return retry_decorator(func)(*args, **kwargs)
            except Exception as e:
                logger.error(f"All retries exhausted for {func.__name__}: {e}")
                if fallback:
                    logger.info(f"Attempting fallback for {func.__name__}")
                    try:
                        return fallback(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        raise e  # Raise original error
                raise
        
        return wrapper
    return decorator


# Pre-configured retry decorators for common use cases

# HTTP API calls - retry on network errors and 5xx
http_retry = async_retry(
    config=RetryConfig(
        max_attempts=3,
        initial_wait=1.0,
        max_wait=10.0
    )
)

# LLM API calls - more aggressive retry
llm_retry = async_retry(
    config=RetryConfig(
        max_attempts=3,
        initial_wait=2.0,
        max_wait=30.0
    )
)

# Database operations - quick retry
db_retry = async_retry(
    config=RetryConfig(
        max_attempts=3,
        initial_wait=0.5,
        max_wait=5.0
    )
)

