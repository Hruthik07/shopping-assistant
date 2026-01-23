"""Circuit breaker implementation for resilient API calls."""

import time
import json as json_module
from typing import Callable, Any, Optional, TypeVar
from functools import wraps
from src.analytics.logger import logger
from src.utils.debug_log import file_debug_log
import asyncio
from collections import defaultdict
from time import time as time_func
from enum import Enum


# #region debug instrumentation
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    try:
        file_debug_log(location, message, data, hypothesis_id=hypothesis_id)
    except Exception:
        pass


# #endregion

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: type = Exception,
        name: Optional[str] = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name


class AsyncCircuitBreaker:
    """Async circuit breaker implementation."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

    def _should_attempt(self) -> bool:
        """Check if request should be attempted."""
        # #region debug instrumentation
        _debug_log(
            "circuit_breaker.py:46",
            "_should_attempt called",
            {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
                "name": self.config.name,
            },
            "F",
        )
        # #endregion
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (
                self.last_failure_time
                and (time_func() - self.last_failure_time) >= self.config.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.config.name} entering HALF_OPEN state")
                # #region debug instrumentation
                _debug_log(
                    "circuit_breaker.py:56",
                    "Circuit breaker entering HALF_OPEN",
                    {"name": self.config.name, "recovery_timeout": self.config.recovery_timeout},
                    "F",
                )
                # #endregion
                return True
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:59",
                "Circuit breaker OPEN - rejecting",
                {
                    "name": self.config.name,
                    "time_since_failure": (
                        time_func() - self.last_failure_time if self.last_failure_time else None
                    ),
                },
                "F",
            )
            # #endregion
            return False

        # HALF_OPEN state - allow one request to test
        # #region debug instrumentation
        _debug_log(
            "circuit_breaker.py:64",
            "Circuit breaker HALF_OPEN - allowing test",
            {"name": self.config.name},
            "F",
        )
        # #endregion
        return True

    def _record_success(self):
        """Record successful request."""
        # #region debug instrumentation
        _debug_log(
            "circuit_breaker.py:63",
            "_record_success called",
            {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "name": self.config.name,
            },
            "F",
        )
        # #endregion
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 1:  # One success in half-open closes circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.config.name} CLOSED after recovery")
                # #region debug instrumentation
                _debug_log(
                    "circuit_breaker.py:70",
                    "Circuit breaker CLOSED after recovery",
                    {"name": self.config.name},
                    "F",
                )
                # #endregion
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:75",
                "Success in CLOSED state - reset failure count",
                {"name": self.config.name},
                "F",
            )
            # #endregion

    def _record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time_func()
        # #region debug instrumentation
        _debug_log(
            "circuit_breaker.py:74",
            "_record_failure called",
            {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "threshold": self.config.failure_threshold,
                "name": self.config.name,
            },
            "F",
        )
        # #endregion

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker {self.config.name} OPENED after {self.failure_count} failures"
            )
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:82",
                "Circuit breaker OPENED",
                {"name": self.config.name, "failure_count": self.failure_count},
                "F",
            )
            # #endregion

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open, go back to open
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker {self.config.name} re-OPENED after failure in HALF_OPEN"
            )
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:88",
                "Circuit breaker re-OPENED from HALF_OPEN",
                {"name": self.config.name},
                "F",
            )
            # #endregion

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        # #region debug instrumentation
        _debug_log(
            "circuit_breaker.py:90",
            "Circuit breaker call started",
            {
                "name": self.config.name,
                "state": self.state.value,
                "func_name": func.__name__ if hasattr(func, "__name__") else "unknown",
            },
            "F",
        )
        # #endregion
        if not self._should_attempt():
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:94",
                "Circuit breaker rejecting call",
                {"name": self.config.name, "state": self.state.value},
                "F",
            )
            # #endregion
            raise Exception(
                f"Circuit breaker {self.config.name} is OPEN. "
                f"Service unavailable. Retry after {self.config.recovery_timeout}s"
            )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:103",
                "Circuit breaker call succeeded",
                {"name": self.config.name, "result_type": type(result).__name__},
                "F",
            )
            # #endregion
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:108",
                "Circuit breaker call failed",
                {"name": self.config.name, "error": str(e), "error_type": type(e).__name__},
                "F",
            )
            # #endregion
            raise
        except Exception as e:
            # Catch any other exceptions and record as failure
            self._record_failure()
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:165",
                "Circuit breaker call failed (unexpected exception)",
                {"name": self.config.name, "error": str(e), "error_type": type(e).__name__},
                "F",
            )
            # #endregion
            raise


# Global circuit breaker instances
_circuit_breakers: dict[str, AsyncCircuitBreaker] = defaultdict(lambda: None)


def async_circuit_breaker(
    config: Optional[CircuitBreakerConfig] = None, fallback: Optional[Callable] = None
):
    """
    Decorator for async functions with circuit breaker pattern.

    Args:
        config: Circuit breaker configuration
        fallback: Fallback function when circuit is open
    """
    if config is None:
        config = CircuitBreakerConfig()

    # Get or create circuit breaker instance
    cb_name = config.name or "default"
    # #region debug instrumentation
    _debug_log(
        "circuit_breaker.py:126",
        "Creating/getting circuit breaker",
        {"cb_name": cb_name, "exists": _circuit_breakers[cb_name] is not None},
        "F",
    )
    # #endregion
    if _circuit_breakers[cb_name] is None:
        _circuit_breakers[cb_name] = AsyncCircuitBreaker(config)
        # #region debug instrumentation
        _debug_log(
            "circuit_breaker.py:129",
            "Created new circuit breaker",
            {
                "cb_name": cb_name,
                "failure_threshold": config.failure_threshold,
                "recovery_timeout": config.recovery_timeout,
            },
            "F",
        )
        # #endregion
    cb = _circuit_breakers[cb_name]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:135",
                "Circuit breaker wrapper called",
                {
                    "func_name": func.__name__,
                    "cb_name": cb_name,
                    "has_fallback": fallback is not None,
                },
                "F",
            )
            # #endregion
            try:
                result = await cb.call(func, *args, **kwargs)
                # #region debug instrumentation
                _debug_log(
                    "circuit_breaker.py:139",
                    "Circuit breaker wrapper success",
                    {"func_name": func.__name__, "result_type": type(result).__name__},
                    "F",
                )
                # #endregion
                return result
            except Exception as e:
                logger.warning(
                    f"Circuit breaker {config.name or func.__name__} blocked or failed: {e}"
                )
                # #region debug instrumentation
                _debug_log(
                    "circuit_breaker.py:146",
                    "Circuit breaker wrapper exception",
                    {
                        "func_name": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "has_fallback": fallback is not None,
                    },
                    "F",
                )
                # #endregion
                if fallback:
                    logger.info(f"Using fallback for {func.__name__}")
                    try:
                        fallback_result = await fallback(*args, **kwargs)
                        # #region debug instrumentation
                        _debug_log(
                            "circuit_breaker.py:152",
                            "Fallback succeeded",
                            {"func_name": func.__name__},
                            "F",
                        )
                        # #endregion
                        return fallback_result
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        # #region debug instrumentation
                        _debug_log(
                            "circuit_breaker.py:157",
                            "Fallback also failed",
                            {"func_name": func.__name__, "fallback_error": str(fallback_error)},
                            "F",
                        )
                        # #endregion
                        raise e
                raise

        return wrapper

    return decorator


def sync_circuit_breaker(
    config: Optional[CircuitBreakerConfig] = None, fallback: Optional[Callable] = None
):
    """
    Decorator for sync functions with circuit breaker pattern.

    Args:
        config: Circuit breaker configuration
        fallback: Fallback function when circuit is open
    """
    if config is None:
        config = CircuitBreakerConfig()

    # #region debug instrumentation
    _debug_log(
        "circuit_breaker.py:164",
        "sync_circuit_breaker called",
        {"name": config.name, "has_fallback": fallback is not None},
        "G",
    )
    # #endregion

    # Create circuit breaker instance (using our async implementation for sync too)
    cb_name = config.name or "default_sync"
    if _circuit_breakers[cb_name] is None:
        _circuit_breakers[cb_name] = AsyncCircuitBreaker(config)
        # #region debug instrumentation
        _debug_log(
            "circuit_breaker.py:171", "Created sync circuit breaker", {"cb_name": cb_name}, "G"
        )
        # #endregion
    cb = _circuit_breakers[cb_name]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # #region debug instrumentation
            _debug_log(
                "circuit_breaker.py:178",
                "Sync circuit breaker wrapper called",
                {"func_name": func.__name__, "cb_name": cb_name},
                "G",
            )
            # #endregion
            try:
                # Wrap sync function to make it async-compatible for circuit breaker
                async def async_func_wrapper():
                    return await asyncio.to_thread(func, *args, **kwargs)

                # Call through circuit breaker (which expects async function)
                result = await cb.call(async_func_wrapper)
                # #region debug instrumentation
                _debug_log(
                    "circuit_breaker.py:186",
                    "Sync circuit breaker wrapper success",
                    {"func_name": func.__name__},
                    "G",
                )
                # #endregion
                return result
            except Exception as e:
                logger.warning(
                    f"Circuit breaker {config.name or func.__name__} is open or failed: {e}"
                )
                # #region debug instrumentation
                _debug_log(
                    "circuit_breaker.py:193",
                    "Sync circuit breaker wrapper exception",
                    {
                        "func_name": func.__name__,
                        "error": str(e),
                        "has_fallback": fallback is not None,
                    },
                    "G",
                )
                # #endregion
                if fallback:
                    logger.info(f"Using fallback for {func.__name__}")
                    try:
                        if asyncio.iscoroutinefunction(fallback):
                            fallback_result = await fallback(*args, **kwargs)
                        else:
                            fallback_result = await asyncio.to_thread(fallback, *args, **kwargs)
                        # #region debug instrumentation
                        _debug_log(
                            "circuit_breaker.py:201",
                            "Sync fallback succeeded",
                            {"func_name": func.__name__},
                            "G",
                        )
                        # #endregion
                        return fallback_result
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        # #region debug instrumentation
                        _debug_log(
                            "circuit_breaker.py:207",
                            "Sync fallback failed",
                            {"func_name": func.__name__, "fallback_error": str(fallback_error)},
                            "G",
                        )
                        # #endregion
                        raise e
                raise

        return wrapper

    return decorator


# Pre-configured circuit breakers for common use cases

# HTTP API circuit breaker
http_circuit_breaker = async_circuit_breaker(
    config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30, name="http_api")
)

# LLM API circuit breaker
llm_circuit_breaker = async_circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5, recovery_timeout=60, name="llm_api"  # Longer recovery for LLM
    )
)

# Database circuit breaker
db_circuit_breaker = async_circuit_breaker(
    config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30, name="database")
)
