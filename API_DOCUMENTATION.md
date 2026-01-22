# API Documentation

## Global Customer-First Product Finder API

### Base URL
```
http://localhost:3565
```

## Endpoints

### Health & Status

#### GET `/api/health/liveness`
Check if the service is alive.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-19T18:00:00Z"
}
```

#### GET `/api/health/readiness`
Check if the service is ready to handle requests.

**Response:**
```json
{
  "status": "ready",
  "database": "connected",
  "cache": "connected"
}
```

### Chat Endpoint

#### POST `/api/chat`
Main chat endpoint for product queries.

**Request:**
```json
{
  "message": "Find me wireless headphones under $100",
  "session_id": "optional_session_id",
  "user_id": 123,
  "persona": "friendly",
  "tone": "casual"
}
```

**Response:**
```json
{
  "response": "Here are some great options...",
  "session_id": "generated_session_id",
  "tools_used": ["search_products"],
  "products": [...],
  "latency_ms": 1234
}
```

### Product Search

#### GET `/api/products/search`
Direct product search endpoint.

**Query Parameters:**
- `q` or `query` (required): Search query
- `category` (optional): Filter by category
- `limit` or `max_results` (optional): Max results (default: 10)
- `min_price` (optional): Minimum price filter
- `max_price` (optional): Maximum price filter

**Example:**
```
GET /api/products/search?q=wireless+headphones&max_price=100&limit=5
```

**Response:**
```json
{
  "query": "wireless headphones",
  "category": null,
  "results": [
    {
      "id": "prod_123",
      "name": "Sony WH-1000XM4",
      "price": 89.99,
      "currency": "USD",
      "shipping_cost": 0.0,
      "retailer": "Amazon",
      "deal_info": {
        "is_deal": true,
        "deal_type": "significant_drop",
        "deal_badge": "Save 15%",
        "savings_percent": 15.2,
        "is_limited_time": false
      },
      "coupon_info": {
        "has_coupon": false
      },
      "price_comparison": {
        "retailer_count": 3,
        "best_price": 89.99,
        "savings_percent": 12.5
      },
      "customer_value": {
        "score": 0.85,
        "total_cost": 89.99
      }
    }
  ],
  "count": 5
}
```

### Metrics Endpoints

#### GET `/api/metrics`
Get general application metrics.

**Response:**
```json
{
  "timestamp": 1705680000,
  "cache": {
    "hits": 1234,
    "misses": 567,
    "hit_rate": 68.5
  },
  "system": {
    "cpu_percent": 45.2,
    "memory_mb": 512.5
  }
}
```

#### GET `/api/metrics/deals`
Get deal detection and price comparison metrics.

**Response:**
```json
{
  "deal_detection": {
    "deals_detected": 450,
    "total_products_analyzed": 2000,
    "deal_detection_rate": 22.5,
    "average_savings_percent": 18.3,
    "total_savings_amount": 12500.50
  },
  "price_comparison": {
    "price_comparisons": 1500,
    "products_with_multiple_retailers": 800,
    "average_retailers_per_product": 2.5,
    "average_price_difference": 15.75
  },
  "api_performance": {
    "total_api_calls": 5000,
    "successful_calls": 4850,
    "failed_calls": 150,
    "success_rate": 97.0
  }
}
```

#### GET `/api/metrics/cache`
Get detailed cache statistics.

#### GET `/api/metrics/latency`
Get latency metrics and percentiles.

#### GET `/api/metrics/cost`
Get LLM cost tracking metrics.

### Error Responses

All endpoints return standard HTTP status codes:
- `200 OK`: Success
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

**Error Response Format:**
```json
{
  "detail": "Error message here",
  "error_code": "ERROR_CODE",
  "timestamp": "2026-01-19T18:00:00Z"
}
```

## Product Data Structure

### Standard Product Format

```json
{
  "id": "unique_product_id",
  "name": "Product Name",
  "description": "Product description",
  "price": 99.99,
  "currency": "USD",
  "original_price": 119.99,
  "shipping_cost": 5.99,
  "image_url": "https://...",
  "product_url": "https://...",
  "rating": 4.5,
  "reviews": 1234,
  "category": "electronics",
  "brand": "Brand Name",
  "availability": true,
  "in_stock": true,
  "retailer": "Amazon",
  "source": "amazon",
  "upc": "123456789012",
  "gtin": "1234567890123",
  "ean": "1234567890123",
  "sku": "SKU123",
  "retailer_options": [
    {
      "retailer": "Amazon",
      "price": 99.99,
      "shipping_cost": 0.0,
      "total_cost": 99.99,
      "product_url": "https://...",
      "availability": true
    },
    {
      "retailer": "Best Buy",
      "price": 104.99,
      "shipping_cost": 5.99,
      "total_cost": 110.98,
      "product_url": "https://...",
      "availability": true
    }
  ],
  "deal_info": {
    "is_deal": true,
    "deal_type": "significant_drop",
    "deal_badge": "Save 15%",
    "savings_percent": 15.2,
    "savings_amount": 18.00,
    "price_trend": "decreasing",
    "is_lowest_price": true,
    "is_limited_time": false,
    "is_seasonal": false
  },
  "coupon_info": {
    "has_coupon": true,
    "coupon_code": "SAVE15",
    "discounted_price": 84.99,
    "savings_percent": 15.0,
    "savings_amount": 15.00
  },
  "price_comparison": {
    "best_price": 99.99,
    "best_retailer": "Amazon",
    "worst_price": 110.98,
    "worst_retailer": "Best Buy",
    "savings": 10.99,
    "savings_percent": 9.9,
    "retailer_count": 2
  },
  "customer_value": {
    "score": 0.85,
    "breakdown": {
      "price_score": 0.90,
      "shipping_score": 1.0,
      "discount_score": 0.75,
      "rating_score": 0.90,
      "deal_score": 0.80
    },
    "total_cost": 99.99,
    "weights": {
      "price": 0.4,
      "shipping": 0.2,
      "discount": 0.2,
      "rating": 0.1,
      "deal": 0.1
    }
  },
  "rank": 1,
  "ranking_explanation": "High customer value score • Save 15% vs average price • Best price among 2 retailers"
}
```

## Rate Limiting

Default rate limit: 60 requests per minute per IP.

Rate limit headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets

## Authentication

Currently, the API uses session-based authentication. Future versions may support API keys.

## WebSocket Support

### WebSocket Endpoint: `/ws`

Connect for streaming responses:

```javascript
const ws = new WebSocket('ws://localhost:3565/ws');
ws.send(JSON.stringify({
  message: "Find me headphones",
  session_id: "session_123"
}));
```

## Examples

### Python

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:3565/api/chat",
        json={
            "message": "Find me wireless headphones under $100",
            "session_id": "my_session"
        }
    )
    data = response.json()
    print(data["response"])
```

### cURL

```bash
curl -X POST http://localhost:3565/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find me wireless headphones under $100"
  }'
```

## Support

For API issues or questions, check:
- Health endpoint: `/api/health/liveness`
- Metrics: `/api/metrics`
- Logs: `logs/app.log`
