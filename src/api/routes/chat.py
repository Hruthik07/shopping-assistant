"""Chat API routes."""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.api.schemas import ChatMessage, ChatResponse
from src.agent.shopping_agent import get_shopping_agent
from src.memory.session_manager import session_manager
from src.database.db import get_db
from src.analytics.logger import logger
from src.analytics.latency_tracker import latency_tracker
from src.analytics.performance_monitor import performance_monitor
from src.utils.guardrails import guardrails, GuardrailViolation
from src.utils.cache import cache_service
from src.analytics.error_tracker import error_tracker
import time

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    db: Session = Depends(get_db)
):
    """Process a chat message with latency tracking and guardrails."""
    request_id = latency_tracker.generate_request_id()
    start_time = time.time()  # Initialize start time for latency tracking
    
    try:
        # Guardrail: Validate query
        is_valid, error_msg = guardrails.validate_query(message.message)
        if not is_valid:
            logger.warning(f"Query validation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Guardrail: Validate session ID
        if message.session_id:
            is_valid, error_msg = guardrails.validate_session_id(message.session_id)
            if not is_valid:
                logger.warning(f"Session ID validation failed: {error_msg}")
                # Don't block, just create new session
                message.session_id = None
        
        # Track session management
        with latency_tracker.track_component("session_management", request_id):
            session_id = message.session_id or session_manager.get_or_create_session()
        
        # Process query (already has internal latency tracking)
        agent = get_shopping_agent()
        result = await agent.process_query(
            query=message.message,
            session_id=session_id,
            persona=message.persona or "friendly",
            tone=message.tone or "warm"
        )
        
        # Guardrail: Validate products
        products = result.get("products", [])
        if products:
            is_valid, error_msg = guardrails.validate_product_data(products)
            if not is_valid:
                logger.warning(f"Product validation issue: {error_msg}")
                # Limit products instead of failing
                result["products"] = products[:guardrails.MAX_PRODUCTS_PER_RESPONSE]
        
        # Guardrail: Sanitize response (with product context for faithfulness)
        result["response"] = guardrails.sanitize_response_with_products(
            result.get("response", ""),
            result.get("products", []) or []
        )
        
        # Add request_id to result if not already present
        if "request_id" not in result:
            result["request_id"] = request_id
        
        # Get cache stats for monitoring
        cache_stats = cache_service.get_cache_stats()
        
        # Record total latency for performance monitoring
        total_latency = time.time() - start_time
        performance_monitor.record_latency(total_latency, request_id=request_id)
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            tools_used=result.get("tools_used", []),
            products=result.get("products", []),
            request_id=result.get("request_id", request_id),
            latency_breakdown=result.get("latency_breakdown", {}),
            cache_stats=cache_stats  # Add cache stats to response
        )
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in chat endpoint: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except TimeoutError as e:
        # Handle timeout errors
        error_tracker.record_error(
            "timeout",
            f"Request timeout: {str(e)}",
            {"request_id": request_id, "query": message.message[:100]}
        )
        logger.error(f"Timeout in chat endpoint: {e}")
        raise HTTPException(status_code=504, detail="Request timeout. Please try again.")
    except Exception as e:
        # Handle all other errors
        error_tracker.record_error(
            "server_error",
            f"Internal server error: {str(e)}",
            {"request_id": request_id, "error_type": type(e).__name__}
        )
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    from src.memory.conversation_store import conversation_store
    
    # Guardrail: Validate session ID
    is_valid, error_msg = guardrails.validate_session_id(session_id)
    if not is_valid:
        logger.warning(f"Invalid session ID in history request: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        history = await conversation_store.get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except ValueError as e:
        logger.warning(f"Validation error getting chat history: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics including hit rate."""
    try:
        internal_stats = cache_service.get_cache_stats()
        redis_stats = await cache_service.get_stats()
        return {
            "internal": internal_stats,
            "redis": redis_stats.get("redis") if isinstance(redis_stats, dict) else None,
            "enabled": cache_service.enabled
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics.")

