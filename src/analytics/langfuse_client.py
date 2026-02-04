"""Langfuse client for monitoring and tracing."""

from typing import Optional, Dict, Any, List
from langfuse import Langfuse
from src.utils.config import settings
from src.analytics.logger import logger


class LangfuseClient:
    """Langfuse client wrapper for tracing and monitoring."""

    def __init__(self):
        """Initialize Langfuse client if enabled."""
        self.enabled = settings.langfuse_enabled
        self.client: Optional[Langfuse] = None

        if self.enabled:
            try:
                if settings.langfuse_public_key and settings.langfuse_secret_key:
                    self.client = Langfuse(
                        public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key,
                        host=settings.langfuse_host,
                    )
                    # SDK compatibility: older versions expose .trace/.span/.generation; newer versions use
                    # start_span/start_generation + trace_context. Detect once and adapt.
                    self._old_api = hasattr(self.client, "trace")
                    logger.info(
                        f"Langfuse client initialized for project: {settings.langfuse_project_name}"
                    )
                else:
                    logger.warning("Langfuse keys not configured, disabling Langfuse")
                    self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse client: {e}")
                self.enabled = False
        else:
            logger.info("Langfuse is disabled in configuration")

    def trace(
        self,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Create a trace in Langfuse."""
        if not self.enabled or not self.client:
            return None

        try:
            # Old SDK API
            if getattr(self, "_old_api", False):
                return self.client.trace(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata or {},
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags or [],
                )

            # New SDK API (trace_id is 32-char hex). We'll create a lightweight root span to materialize the trace.
            trace_id = self.client.create_trace_id(seed=session_id or user_id)
            root = self.client.start_span(
                trace_context={"trace_id": trace_id},
                name=name,
                input=input,
                output=output,
                metadata=metadata or {},
            )
            try:
                root.end()
            except Exception:
                pass
            return {"id": trace_id}
        except Exception as e:
            logger.error(f"Failed to create Langfuse trace: {e}")
            return None

    def span(
        self,
        trace_id: str,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a span within a trace. Returns a context manager for proper duration tracking.

        Usage:
            with langfuse_client.span(trace_id, "operation_name", input={...}) as span:
                # Do work here
                span.update(output={...})
        """
        if not self.enabled or not self.client:
            return _NullSpanContext()

        return _SpanContext(
            client=self.client,
            old_api=getattr(self, "_old_api", False),
            trace_id=trace_id,
            name=name,
            input=input,
            output=output,
            metadata=metadata or {},
        )

    def generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a generation (LLM call) within a trace."""
        if not self.enabled or not self.client:
            return None

        try:
            if getattr(self, "_old_api", False):
                return self.client.generation(
                    trace_id=trace_id,
                    name=name,
                    model=model,
                    input=input,
                    output=output,
                    usage=usage,
                    metadata=metadata or {},
                )

            gen = self.client.start_generation(
                trace_context={"trace_id": trace_id},
                name=name,
                model=model,
                input=input,
                output=output,
                metadata=metadata or {},
                usage_details=usage,
            )
            try:
                gen.end()
            except Exception:
                pass
            return {"id": getattr(gen, "id", None), "trace_id": trace_id}
        except Exception as e:
            logger.error(f"Failed to create Langfuse generation: {e}")
            return None

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        data_type: str = "NUMERIC",
    ):
        """Add a score to a trace.

        Args:
            trace_id: Langfuse trace ID
            name: Score name (must match ScoreConfig name if using configs)
            value: Score value (0-1 for numeric, or boolean/categorical)
            comment: Optional comment/explanation
            data_type: Score data type - "NUMERIC", "BOOLEAN", or "CATEGORICAL"
        """
        if not self.enabled or not self.client:
            return None

        try:
            # Use create_score() - the correct Langfuse SDK method
            # Note: create_score() returns None but the score IS created successfully
            if hasattr(self.client, "create_score"):
                self.client.create_score(
                    trace_id=trace_id, name=name, value=value, comment=comment, data_type=data_type
                )
                # create_score() returns None, but score is created successfully
                # Return a dict to indicate success
                return {"success": True, "trace_id": trace_id, "name": name, "value": value}
            # Fallback: try score_current_trace if we're in trace context
            elif hasattr(self.client, "score_current_trace"):
                logger.warning(
                    f"Using score_current_trace fallback for {name} - trace_id may not be used"
                )
                return self.client.score_current_trace(name=name, value=value, comment=comment)
            else:
                logger.warning(
                    "Langfuse client has no score method (create_score or score_current_trace)"
                )
                return None
        except Exception as e:
            logger.error(f"Failed to create Langfuse score '{name}' for trace {trace_id}: {e}")
            return None

    def update_trace(
        self,
        trace_id: str,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update an existing trace with output and metadata."""
        if not self.enabled or not self.client or not trace_id:
            return None

        try:
            if output or metadata:
                # Use span as context manager for proper tracking
                with self.span(
                    trace_id=trace_id, name="final_output", output=output, metadata=metadata
                ):
                    pass  # Span is created and immediately ended with output/metadata
            return True
        except Exception as e:
            logger.error(f"Failed to update Langfuse trace: {e}")
            return None

    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse events: {e}")


class _SpanContext:
    """Context manager for Langfuse spans that properly tracks duration.

    Supports both sync and async operations. The span starts when entering
    the context and ends when exiting, properly tracking the duration of operations.
    """

    def __init__(
        self,
        client: Langfuse,
        old_api: bool,
        trace_id: str,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.old_api = old_api
        self.trace_id = trace_id
        self.name = name
        self.input = input
        self.output = output
        self.metadata = metadata
        self.span_obj = None

    def __enter__(self):
        """Start the span when entering the context."""
        try:
            if self.old_api:
                self.span_obj = self.client.span(
                    trace_id=self.trace_id,
                    name=self.name,
                    input=self.input,
                    output=None,  # Will be set on exit
                    metadata=self.metadata or {},
                )
            else:
                self.span_obj = self.client.start_span(
                    trace_context={"trace_id": self.trace_id},
                    name=self.name,
                    input=self.input,
                    output=None,  # Will be set on exit
                    metadata=self.metadata or {},
                )
        except Exception as e:
            logger.error(f"Failed to start Langfuse span '{self.name}': {e}")
            self.span_obj = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span when exiting the context."""
        if self.span_obj:
            try:
                # Update output if provided
                if self.output:
                    if hasattr(self.span_obj, "update"):
                        self.span_obj.update(output=self.output)
                    elif hasattr(self.span_obj, "end"):
                        # For new API, we can set output before ending
                        pass

                # End the span
                if hasattr(self.span_obj, "end"):
                    self.span_obj.end(output=self.output)
                elif hasattr(self.span_obj, "update"):
                    # Old API might need explicit update
                    self.span_obj.update(output=self.output)
            except Exception as e:
                logger.debug(f"Failed to end Langfuse span '{self.name}': {e}")
        return False  # Don't suppress exceptions

    async def __aenter__(self):
        """Async context manager entry - same as sync."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - same as sync."""
        return self.__exit__(exc_type, exc_val, exc_tb)

    def update(
        self, output: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None
    ):
        """Update span output or metadata while it's active."""
        if self.span_obj:
            try:
                if output:
                    self.output = output
                if metadata:
                    self.metadata = {**(self.metadata or {}), **metadata}

                if hasattr(self.span_obj, "update"):
                    self.span_obj.update(output=output, metadata=metadata)
            except Exception as e:
                logger.debug(f"Failed to update Langfuse span '{self.name}': {e}")


class _NullSpanContext:
    """Null context manager when Langfuse is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def update(
        self, output: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None
    ):
        pass


# Global Langfuse client instance
langfuse_client = LangfuseClient()
