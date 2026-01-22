"""WebSocket support for streaming responses."""
import json
from fastapi import WebSocket, WebSocketDisconnect
from src.agent.shopping_agent import get_shopping_agent
from src.memory.session_manager import session_manager
from src.analytics.logger import logger
from src.utils.guardrails import guardrails
from langchain_core.messages import HumanMessage


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection gracefully."""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        except (ValueError, AttributeError) as e:
            logger.debug(f"Error disconnecting websocket (may already be disconnected): {e}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to a specific connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            # Disconnect on send failure
            self.disconnect(websocket)
            raise
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming chat with proper error handling."""
    await manager.connect(websocket)
    session_id = None
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                
                # Guardrail: Validate query
                is_valid, error_msg = guardrails.validate_query(data)
                if not is_valid:
                    logger.warning(f"WebSocket query validation failed: {error_msg}")
                    await manager.send_personal_message(
                        f"Error: {error_msg}",
                        websocket
                    )
                    continue
                
                # Get or create session
                if not session_id:
                    session_id = session_manager.get_or_create_session()
                
                # Process query with streaming
                try:
                    agent = get_shopping_agent()
                    
                    # Try streaming first
                    try:
                        # Build messages for streaming
                        from src.memory.conversation_store import conversation_store
                        history = await conversation_store.get_conversation_history(session_id)
                        messages = agent._build_messages(history, data)
                        messages.append(HumanMessage(content=data))
                        
                        # Stream response token by token
                        full_response = ""
                        async for chunk in agent.stream_response(messages):
                            full_response += chunk
                            # Send chunk to client
                            await manager.send_personal_message(
                                json.dumps({"type": "chunk", "content": chunk}),
                                websocket
                            )
                        
                        # Send completion signal
                        await manager.send_personal_message(
                            json.dumps({"type": "complete", "response": full_response}),
                            websocket
                        )
                    except Exception as stream_error:
                        logger.debug(f"Streaming failed, falling back to non-streaming: {stream_error}")
                        # Fallback to non-streaming
                        result = await agent.process_query(
                            query=data,
                            session_id=session_id
                        )
                        
                        # Guardrail: Sanitize response (with product context for faithfulness)
                        response = guardrails.sanitize_response_with_products(
                            result.get("response", ""),
                            result.get("products", []) or []
                        )
                        
                        # Send response
                        await manager.send_personal_message(
                            json.dumps({"type": "complete", "response": response}),
                            websocket
                        )
                    
                except Exception as agent_error:
                    logger.error(f"Error processing query in WebSocket: {agent_error}", exc_info=True)
                    error_message = "I'm sorry, I encountered an error processing your request. Please try again."
                    await manager.send_personal_message(error_message, websocket)
                    
            except WebSocketDisconnect:
                # Client disconnected, break the loop
                break
            except Exception as message_error:
                logger.error(f"Error handling WebSocket message: {message_error}", exc_info=True)
                try:
                    await manager.send_personal_message(
                        "An error occurred. Please try again.",
                        websocket
                    )
                except:
                    # Connection may be broken, break the loop
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}", exc_info=True)
    finally:
        manager.disconnect(websocket)
        logger.info(f"WebSocket connection closed: {session_id}")

