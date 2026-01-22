"""Web search MCP tools."""
from typing import Dict, Any, Optional
from src.mcp.mcp_client import MCPTool, tool_registry
from src.utils.config import settings
from src.analytics.logger import logger


class WebSearchTool(MCPTool):
    """Search the web for product reviews and information."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for product reviews, comparisons, and additional information. Useful for finding external reviews and product details."
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Execute web search."""
        if not settings.tavily_api_key:
            return {
                "query": query,
                "results": [],
                "message": "Tavily API key not configured"
            }
        
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)
            
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced"
            )
            
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:500],  # Truncate
                    "score": item.get("score", 0)
                })
            
            return {
                "query": query,
                "results_count": len(results),
                "results": results,
                "message": f"Found {len(results)} web results"
            }
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {
                "query": query,
                "results": [],
                "error": str(e)
            }


class ProductReviewSearchTool(MCPTool):
    """Search specifically for product reviews."""
    
    def __init__(self):
        super().__init__(
            name="search_product_reviews",
            description="Search the web for reviews of a specific product. Returns external reviews and ratings."
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Name of the product to search reviews for"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of review results (default: 5)",
                    "default": 5
                }
            },
            "required": ["product_name"]
        }
    
    async def execute(self, product_name: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for product reviews."""
        query = f"{product_name} review rating"
        web_search = WebSearchTool()
        return await web_search.execute(query, max_results)


# Register search tools
web_search_tool = WebSearchTool()
product_review_search_tool = ProductReviewSearchTool()

tool_registry.register(web_search_tool)
tool_registry.register(product_review_search_tool)

