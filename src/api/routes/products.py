"""Product API routes."""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.api.product_fetcher import product_fetcher
from src.analytics.logger import logger

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get("/search")
async def search_products(
    q: Optional[str] = Query(None, description="Search query (preferred param: q)"),
    query: Optional[str] = Query(None, description="Search query (alias for q)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: Optional[int] = Query(None, ge=1, le=50, description="Maximum results (alias: max_results)"),
    max_results: Optional[int] = Query(None, ge=1, le=50, description="Maximum results (alias for limit)")
):
    """Search for products."""
    try:
        q_effective = (q or query or "").strip()
        if not q_effective:
            raise HTTPException(status_code=422, detail="Missing required query parameter: q (or query)")

        limit_effective = max_results or limit or 10

        # Use API to fetch products
        api_products = await product_fetcher.search_products(
            query=q_effective,
            num_results=limit_effective,
            use_google_shopping=True
        )
        
        # Filter by category if specified
        if category:
            api_products = [p for p in api_products if p.get("category", "").lower() == category.lower()]
        
        return {
            "query": q_effective,
            "category": category,
            "results": api_products[:limit_effective],
            "count": len(api_products)
        }
    except Exception as e:
        logger.error(f"Error searching products: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{product_id}")
async def get_product(product_id: str):
    """Get product details by ID."""
    try:
        from src.data.document_loader import document_loader
        
        # Get product
        product = await document_loader.get_product_by_id(product_id)
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return {
            "product": product
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

