"""Amazon Product Advertising API data source."""

import httpx
import hmac
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from src.services.product_data_source import ProductDataSource
from src.utils.config import settings
from src.analytics.logger import logger


class AmazonDataSource(ProductDataSource):
    """Amazon Product Advertising API data source."""

    def __init__(self):
        self.api_key = settings.amazon_api_key
        self.secret_key = settings.amazon_secret_key
        self.associate_tag = settings.amazon_associate_tag
        # Amazon PA-API 5.0 endpoint
        self.endpoint = "https://webservices.amazon.com/paapi5/searchitems"

    def get_source_name(self) -> str:
        return "amazon"

    def is_available(self) -> bool:
        return bool(self.api_key and self.secret_key and self.associate_tag)

    def _sign_request(self, payload: Dict[str, Any], timestamp: str) -> Dict[str, str]:
        """Generate AWS Signature V4 for PA-API request.

        Args:
            payload: Request payload
            timestamp: ISO 8601 timestamp

        Returns:
            Dictionary with Authorization header
        """
        # AWS Signature V4 implementation for PA-API
        # This is a simplified version - full implementation would use boto3 or proper signing

        # For now, return basic headers - full signing requires:
        # 1. Create canonical request
        # 2. Create string to sign
        # 3. Calculate signature
        # 4. Build authorization header

        # Note: Full implementation would require:
        # - Proper canonical request formatting
        # - HMAC-SHA256 signing
        # - Credential scope
        # - Signed headers

        # Placeholder - returns headers that would work with proper signing
        return {
            "Content-Type": "application/json; charset=utf-8",
            "X-Amz-Target": "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems",
            "X-Amz-Date": timestamp,
        }

    async def search_products(
        self,
        query: str,
        num_results: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search products using Amazon Product Advertising API."""
        if not self.is_available():
            return []

        try:
            # Amazon PA-API 5.0 requires AWS Signature V4
            # Full implementation requires proper AWS signing library
            # For production, consider using: boto3, requests-aws4auth, or paapi5-python-sdk

            # Build request payload
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

            payload = {
                "PartnerTag": self.associate_tag,
                "PartnerType": "Associates",
                "Keywords": query,
                "Resources": [
                    "ItemInfo.Title",
                    "ItemInfo.ByLineInfo",
                    "ItemInfo.Classifications",
                    "ItemInfo.ContentInfo",
                    "ItemInfo.ExternalIds",
                    "ItemInfo.Features",
                    "ItemInfo.ManufactureInfo",
                    "ItemInfo.ProductInfo",
                    "ItemInfo.TechnicalInfo",
                    "ItemInfo.TradeInInfo",
                    "Offers.Listings.Price",
                    "Offers.Listings.Availability",
                    "Offers.Listings.Condition",
                    "Offers.Listings.MerchantInfo",
                    "Offers.Summaries.HighestPrice",
                    "Offers.Summaries.LowestPrice",
                    "Offers.Summaries.OfferCount",
                    "Images.Primary.Large",
                    "Images.Variants.Large",
                    "CustomerReviews.StarRating",
                    "CustomerReviews.Count",
                ],
                "ItemCount": min(num_results, 10),  # PA-API limit is 10
                "SearchIndex": "All",
            }

            # Add price filters if specified
            if min_price or max_price:
                price_range = {}
                if min_price:
                    price_range["Min"] = min_price
                if max_price:
                    price_range["Max"] = max_price
                payload["MinPrice"] = price_range.get("Min")
                payload["MaxPrice"] = price_range.get("Max")

            # Add category filter
            if category:
                # Map category to Amazon SearchIndex
                category_map = {
                    "electronics": "Electronics",
                    "books": "Books",
                    "clothing": "Fashion",
                    "home": "HomeAndKitchen",
                }
                search_index = category_map.get(category.lower(), "All")
                payload["SearchIndex"] = search_index

            # Generate signed headers (simplified - full implementation needed)
            headers = self._sign_request(payload, timestamp)
            headers["Content-Encoding"] = "amz-1.0"

            # Note: Full implementation would include proper Authorization header
            # For now, log that proper signing is needed
            logger.warning(
                "Amazon PA-API requires full AWS Signature V4 implementation. "
                "Consider using paapi5-python-sdk or implementing proper signing. "
                "Returning empty results."
            )

            # Uncomment below when proper signing is implemented:
            # async with httpx.AsyncClient(timeout=10.0) as client:
            #     response = await client.post(
            #         self.endpoint,
            #         headers=headers,
            #         json=payload
            #     )
            #     response.raise_for_status()
            #     data = response.json()
            #
            #     products = []
            #     for item in data.get("SearchResult", {}).get("Items", []):
            #         normalized = self.normalize_product(item)
            #         products.append(normalized)
            #
            #     return products

            return []

        except Exception as e:
            logger.error(f"Error fetching from Amazon API: {e}", exc_info=True)
            return []

    def normalize_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Amazon PA-API product response to standard format.

        PA-API 5.0 response structure:
        {
            "ASIN": "...",
            "ItemInfo": {
                "Title": {"DisplayValue": "..."},
                "ByLineInfo": {"Brand": {"DisplayValue": "..."}},
                ...
            },
            "Offers": {
                "Listings": [{
                    "Price": {"DisplayAmount": "$XX.XX"},
                    "Availability": {"Message": "..."}
                }]
            },
            "Images": {...},
            "CustomerReviews": {...}
        }
        """
        # Extract ASIN
        asin = product.get("ASIN", "")

        # Extract item info
        item_info = product.get("ItemInfo", {})
        title_info = item_info.get("Title", {})
        title = (
            title_info.get("DisplayValue", "") if isinstance(title_info, dict) else str(title_info)
        )

        byline = item_info.get("ByLineInfo", {})
        brand_info = byline.get("Brand", {}) if isinstance(byline, dict) else {}
        brand = brand_info.get("DisplayValue", "") if isinstance(brand_info, dict) else ""

        # Extract price
        offers = product.get("Offers", {})
        listings = offers.get("Listings", []) if isinstance(offers, dict) else []
        price = 0.0
        currency = "USD"
        availability = True

        if listings:
            first_listing = listings[0]
            price_info = first_listing.get("Price", {})
            if isinstance(price_info, dict):
                price_str = price_info.get("DisplayAmount", "$0.00")
                # Parse price string like "$XX.XX"
                try:
                    price = float(price_str.replace("$", "").replace(",", "").strip())
                except (ValueError, AttributeError):
                    price = 0.0

                currency_info = price_info.get("Currency", "USD")
                currency = currency_info if isinstance(currency_info, str) else "USD"

            availability_info = first_listing.get("Availability", {})
            if isinstance(availability_info, dict):
                message = availability_info.get("Message", "")
                availability = "In Stock" in message or "Available" in message

        # Extract images
        images = product.get("Images", {})
        primary = images.get("Primary", {}) if isinstance(images, dict) else {}
        large = primary.get("Large", {}) if isinstance(primary, dict) else {}
        image_url = large.get("URL", "") if isinstance(large, dict) else ""

        # Extract reviews
        reviews = product.get("CustomerReviews", {})
        star_rating = 0.0
        review_count = 0

        if isinstance(reviews, dict):
            star_info = reviews.get("StarRating", {})
            if isinstance(star_info, dict):
                star_rating = float(star_info.get("Value", 0))

            count_info = reviews.get("Count", {})
            if isinstance(count_info, dict):
                review_count = int(count_info.get("Value", 0))

        # Extract external IDs (UPC, EAN, etc.)
        external_ids = item_info.get("ExternalIds", {}) if isinstance(item_info, dict) else {}
        upc = ""
        ean = ""

        if isinstance(external_ids, dict):
            upc_info = external_ids.get("UPCs", {})
            if isinstance(upc_info, dict):
                upc_values = upc_info.get("DisplayValues", [])
                upc = upc_values[0] if upc_values else ""

            ean_info = external_ids.get("EANs", {})
            if isinstance(ean_info, dict):
                ean_values = ean_info.get("DisplayValues", [])
                ean = ean_values[0] if ean_values else ""

        # Build product URL
        product_url = f"https://www.amazon.com/dp/{asin}" if asin else ""

        return {
            "id": f"amazon_{asin}",
            "name": title,
            "description": "",
            "price": price,
            "currency": currency,
            "original_price": None,
            "shipping_cost": 0.0,  # Amazon Prime items have free shipping
            "image_url": image_url,
            "product_url": product_url,
            "rating": star_rating,
            "reviews": review_count,
            "category": "general",
            "brand": brand,
            "availability": availability,
            "in_stock": availability,
            "retailer": "Amazon",
            "source": "amazon",
            "upc": upc,
            "gtin": None,
            "ean": ean,
            "sku": None,
            "metadata": {"asin": asin},
        }
