"""Image analysis MCP tools."""
from typing import Dict, Any, Optional
from src.mcp.mcp_client import MCPTool, tool_registry
from src.utils.config import settings
from src.analytics.logger import logger
import base64
from pathlib import Path


class ImageAnalysisTool(MCPTool):
    """Analyze product images to find similar products."""
    
    def __init__(self):
        super().__init__(
            name="analyze_product_image",
            description="Analyze a product image to identify the product and find similar items. Accepts image URL or base64 encoded image."
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the product image"
                },
                "image_base64": {
                    "type": "string",
                    "description": "Base64 encoded image data (alternative to image_url)"
                }
            }
        }
    
    async def execute(self, image_url: Optional[str] = None, image_base64: Optional[str] = None) -> Dict[str, Any]:
        """Analyze product image."""
        if not settings.openai_api_key:
            return {
                "error": "OpenAI API key not configured",
                "message": "Image analysis requires OpenAI API key"
            }
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            
            # Prepare image
            if image_url:
                image_input = image_url
            elif image_base64:
                image_input = image_base64
            else:
                return {
                    "error": "No image provided",
                    "message": "Please provide either image_url or image_base64"
                }
            
            # Use GPT-4 Vision to analyze image
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify this product. Describe what it is, its key features, and what category it belongs to. Be specific about brand, model, or type if visible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_input if image_url else f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            description = response.choices[0].message.content
            
            return {
                "image_analyzed": True,
                "product_description": description,
                "message": "Image analyzed successfully. Use this description to search for similar products."
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "error": str(e),
                "message": "Failed to analyze image"
            }


class FindSimilarProductsTool(MCPTool):
    """Find similar products based on image analysis."""
    
    def __init__(self):
        super().__init__(
            name="find_similar_products",
            description="Find products similar to a given product description or image analysis result."
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Product description or image analysis result"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of similar products (default: 5)",
                    "default": 5
                }
            },
            "required": ["description"]
        }
    
    async def execute(self, description: str, max_results: int = 5) -> Dict[str, Any]:
        """Find similar products."""
        # Use API to find similar products
        from src.api.product_fetcher import product_fetcher
        
        try:
            products = await product_fetcher.search_products(
                query=description,
                num_results=max_results,
                use_google_shopping=True
            )
            
            return {
                "description": description,
                "results_count": len(products),
                "products": [
                    {
                        "id": p.get("id", ""),
                        "name": p.get("name", ""),
                        "category": p.get("category", ""),
                        "price": p.get("price", ""),
                        "image_url": p.get("image_url", "")
                    }
                    for p in products
                ],
                "message": f"Found {len(products)} similar products"
            }
        except Exception as e:
            logger.error(f"Error finding similar products: {e}")
            return {
                "description": description,
                "results_count": 0,
                "products": [],
                "error": str(e)
            }


# Register image tools
image_analysis_tool = ImageAnalysisTool()
find_similar_products_tool = FindSimilarProductsTool()

tool_registry.register(image_analysis_tool)
tool_registry.register(find_similar_products_tool)

