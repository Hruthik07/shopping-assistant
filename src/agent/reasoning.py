"""Agent reasoning and decision-making logic."""
from typing import Dict, Any, List, Optional
from src.analytics.logger import logger


class AgentReasoner:
    """Handle agent reasoning and tool selection."""
    
    def determine_intent(self, query: str) -> Dict[str, Any]:
        """Determine user intent from query."""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "confidence": 0.5,
            "tools_needed": []
        }
        
        # Search intent - detect explicit search queries
        search_keywords = ["find", "search", "look for", "show me", "what", "where", "give me", "get me", "need", "want", "looking for", "now", "give"]
        
        if any(word in query_lower for word in search_keywords):
            intent["type"] = "search"
            intent["confidence"] = 0.8
            intent["tools_needed"] = ["search_products"]
        # Handle follow-up queries - detect using pattern matching instead of hardcoded keywords
        # Short queries (≤6 words) with refinement indicators are likely search refinements
        elif len(query.split()) <= 6:
            refinement_indicators = ["in", "from", "by", "now", "give", "show", "also", "instead"]
            # Check for capitalized words (likely brand names) or refinement words
            import re
            has_capitalized = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', query))
            has_refinement = any(word in query_lower for word in refinement_indicators)
            
            if has_capitalized or has_refinement:
                intent["type"] = "search"
                intent["confidence"] = 0.7
                intent["tools_needed"] = ["search_products"]
        
        # Question intent
        elif any(word in query_lower for word in ["how", "why", "what is", "tell me about", "explain"]):
            intent["type"] = "question"
            intent["confidence"] = 0.7
            intent["tools_needed"] = ["web_search"]
        
        # Price/availability check
        elif any(word in query_lower for word in ["price", "cost", "available", "in stock", "how much"]):
            intent["type"] = "check"
            intent["confidence"] = 0.8
            intent["tools_needed"] = ["check_product_price", "check_product_availability"]
        
        # Cart operations
        elif any(word in query_lower for word in ["add to cart", "cart", "buy", "purchase"]):
            intent["type"] = "cart"
            intent["confidence"] = 0.9
            intent["tools_needed"] = ["add_to_cart", "get_cart"]
        
        # Comparison
        elif any(word in query_lower for word in ["compare", "difference", "better", "vs", "versus"]):
            intent["type"] = "compare"
            intent["confidence"] = 0.8
            intent["tools_needed"] = ["search_products", "check_product_price"]
        
        # Image search
        elif any(word in query_lower for word in ["image", "photo", "picture", "looks like"]):
            intent["type"] = "image"
            intent["confidence"] = 0.7
            intent["tools_needed"] = ["analyze_product_image", "find_similar_products"]
        
        return intent
    
    def select_tools(self, intent: Dict[str, Any], available_tools: List[str]) -> List[str]:
        """Select appropriate tools based on intent."""
        tools_needed = intent.get("tools_needed", [])
        
        # Filter to only available tools
        selected = [tool for tool in tools_needed if tool in available_tools]
        
        return selected
    
    def should_continue(self, response: Dict[str, Any], max_iterations: int = 3) -> bool:
        """Determine if agent should continue processing."""
        # Check if we have enough information
        if response.get("complete", False):
            return False
        
        # Check iteration limit
        if response.get("iterations", 0) >= max_iterations:
            return False
        
        # Check if we got useful results
        if response.get("results") and len(response.get("results", [])) > 0:
            return False
        
        return True


# Global reasoner
reasoner = AgentReasoner()

