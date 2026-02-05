"""Guardrails for AI shopping assistant - safety, security, and content filtering."""

import os
import re
import time
import json as json_module
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
from src.analytics.logger import logger

GUARDRAILS_DEBUG = os.getenv("GUARDRAILS_DEBUG", "false").lower() == "true"


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    """Guardrail debug logging (opt-in).

    Enabled only when GUARDRAILS_DEBUG=true is set in the environment.
    """
    if not GUARDRAILS_DEBUG:
        return
    try:
        payload = {
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
        }
        logger.debug(f"[guardrails_debug] {json_module.dumps(payload, ensure_ascii=False)}")
    except Exception:
        # Never let debug logging break request handling
        pass


class GuardrailViolation(Exception):
    """Exception raised when a guardrail is violated."""

    def __init__(self, violation_type: str, message: str, severity: str = "medium"):
        self.violation_type = violation_type
        self.message = message
        self.severity = severity
        super().__init__(message)


class ShoppingAssistantGuardrails:
    """Comprehensive guardrails for shopping assistant."""

    # Blocked patterns (harmful, inappropriate, or malicious content)
    BLOCKED_PATTERNS = [
        r"\b(hack|exploit|vulnerability|sql injection|xss)\b",
        r"\b(illegal|stolen|counterfeit|fake)\s+(product|item|goods)",
        r"\b(weapon|gun|knife|ammunition)\b",
        r"\b(drug|narcotic|cocaine|heroin|marijuana)\b",
        r"\b(porn|adult|explicit|nsfw)\b",
        r"<script|javascript:|onerror=|onclick=",
        r"\.\.\/|\.\.\\",  # Path traversal
    ]

    # Suspicious patterns (warn but don't block)
    SUSPICIOUS_PATTERNS = [
        r"\b(free\s+money|get\s+rich|make\s+\$\d+)",
        r"\b(click\s+here|urgent|limited\s+time)",
        r"[A-Z]{10,}",  # Excessive caps
        r"[!]{3,}",  # Excessive exclamation marks
    ]

    # Maximum limits
    MAX_QUERY_LENGTH = 500
    MAX_TOOL_CALLS_PER_REQUEST = 10
    MAX_PRODUCTS_PER_RESPONSE = 50
    MAX_PRICE = 1000000  # $1M max price filter

    def __init__(self):
        self.blocked_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.BLOCKED_PATTERNS
        ]
        self.suspicious_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SUSPICIOUS_PATTERNS
        ]

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query for safety and appropriateness.

        Returns:
            (is_valid, error_message)
        """
        # #region debug instrumentation
        _debug_log(
            "guardrails.py:48",
            "validate_query called",
            {
                "query_length": len(query) if query else 0,
                "query_type": type(query).__name__,
                "query_preview": query[:50] if query else None,
            },
            "K",
        )
        # #endregion
        if not query or not isinstance(query, str):
            # #region debug instrumentation
            _debug_log(
                "guardrails.py:56",
                "validate_query failed - not string or empty",
                {"query": query},
                "K",
            )
            # #endregion
            return False, "Query must be a non-empty string"

        # Length check
        if len(query) > self.MAX_QUERY_LENGTH:
            # #region debug instrumentation
            _debug_log(
                "guardrails.py:61",
                "validate_query failed - too long",
                {"query_length": len(query), "max_length": self.MAX_QUERY_LENGTH},
                "K",
            )
            # #endregion
            return False, f"Query too long. Maximum {self.MAX_QUERY_LENGTH} characters allowed."

        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(query):
                logger.warning(f"Blocked query due to pattern match: {pattern.pattern}")
                # #region debug instrumentation
                _debug_log(
                    "guardrails.py:66",
                    "validate_query blocked - pattern match",
                    {"pattern": pattern.pattern, "query_preview": query[:50]},
                    "K",
                )
                # #endregion
                return (
                    False,
                    "I can't help with that type of request. Please ask about legitimate shopping needs.",
                )

        # Check for suspicious patterns (log but allow)
        for pattern in self.suspicious_patterns:
            if pattern.search(query):
                logger.warning(f"Suspicious pattern detected in query: {pattern.pattern}")
                # #region debug instrumentation
                _debug_log(
                    "guardrails.py:72",
                    "validate_query suspicious pattern",
                    {"pattern": pattern.pattern},
                    "K",
                )
                # #endregion

        # #region debug instrumentation
        _debug_log("guardrails.py:75", "validate_query passed", {"query_length": len(query)}, "K")
        # #endregion
        return True, None

    def validate_price_range(
        self, min_price: Optional[float], max_price: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """Validate price range parameters."""
        # #region debug instrumentation
        _debug_log(
            "guardrails.py:75",
            "validate_price_range called",
            {"min_price": min_price, "max_price": max_price, "max_allowed": self.MAX_PRICE},
            "K",
        )
        # #endregion
        if min_price is not None:
            if min_price < 0:
                # #region debug instrumentation
                _debug_log(
                    "guardrails.py:79",
                    "validate_price_range failed - negative min",
                    {"min_price": min_price},
                    "K",
                )
                # #endregion
                return False, "Minimum price cannot be negative"
            if min_price > self.MAX_PRICE:
                # #region debug instrumentation
                _debug_log(
                    "guardrails.py:83",
                    "validate_price_range failed - min too high",
                    {"min_price": min_price, "max_allowed": self.MAX_PRICE},
                    "K",
                )
                # #endregion
                return False, f"Maximum price limit is ${self.MAX_PRICE:,.2f}"

        if max_price is not None:
            if max_price < 0:
                # #region debug instrumentation
                _debug_log(
                    "guardrails.py:88",
                    "validate_price_range failed - negative max",
                    {"max_price": max_price},
                    "K",
                )
                # #endregion
                return False, "Maximum price cannot be negative"
            if max_price > self.MAX_PRICE:
                # #region debug instrumentation
                _debug_log(
                    "guardrails.py:92",
                    "validate_price_range failed - max too high",
                    {"max_price": max_price, "max_allowed": self.MAX_PRICE},
                    "K",
                )
                # #endregion
                return False, f"Maximum price limit is ${self.MAX_PRICE:,.2f}"

        if min_price is not None and max_price is not None:
            if min_price > max_price:
                # #region debug instrumentation
                _debug_log(
                    "guardrails.py:98",
                    "validate_price_range failed - min > max",
                    {"min_price": min_price, "max_price": max_price},
                    "K",
                )
                # #endregion
                return False, "Minimum price cannot be greater than maximum price"

        # #region debug instrumentation
        _debug_log(
            "guardrails.py:102",
            "validate_price_range passed",
            {"min_price": min_price, "max_price": max_price},
            "K",
        )
        # #endregion
        return True, None

    def validate_tool_usage(
        self, tool_name: str, tool_calls_count: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate tool usage limits."""
        if tool_calls_count >= self.MAX_TOOL_CALLS_PER_REQUEST:
            return (
                False,
                f"Too many tool calls. Maximum {self.MAX_TOOL_CALLS_PER_REQUEST} calls per request.",
            )

        # Tool-specific validations
        if tool_name == "search_products":
            # Additional validations for product search
            pass

        return True, None

    def validate_product_data(self, products: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Validate product data before returning to user."""
        if len(products) > self.MAX_PRODUCTS_PER_RESPONSE:
            logger.warning(
                f"Too many products returned: {len(products)}. Limiting to {self.MAX_PRODUCTS_PER_RESPONSE}"
            )
            return (
                False,
                f"Too many products. Maximum {self.MAX_PRODUCTS_PER_RESPONSE} products per response.",
            )

        # Validate each product
        for product in products:
            # Check for required fields
            if not product.get("name") and not product.get("title"):
                logger.warning("Product missing name/title")
                continue

            # Validate price
            price = product.get("price", 0)
            if isinstance(price, (int, float)):
                if price < 0 or price > self.MAX_PRICE:
                    logger.warning(f"Invalid price in product: {price}")
                    product["price"] = min(max(0, price), self.MAX_PRICE)

            # Sanitize product URLs
            if "product_url" in product or "link" in product:
                url = product.get("product_url") or product.get("link")
                if url and not self._is_safe_url(url):
                    logger.warning(f"Unsafe URL detected: {url}")
                    # Remove unsafe URLs
                    product.pop("product_url", None)
                    product.pop("link", None)

        return True, None

    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe (not malicious)."""
        # #region debug instrumentation
        _debug_log(
            "guardrails.py:138",
            "_is_safe_url called",
            {"url": url[:100] if url else None, "url_type": type(url).__name__},
            "K",
        )
        # #endregion
        if not isinstance(url, str):
            # #region debug instrumentation
            _debug_log(
                "guardrails.py:141",
                "_is_safe_url failed - not string",
                {"url_type": type(url).__name__},
                "K",
            )
            # #endregion
            return False

        # Must start with http:// or https://
        if not url.startswith(("http://", "https://")):
            # #region debug instrumentation
            _debug_log(
                "guardrails.py:146", "_is_safe_url failed - invalid scheme", {"url": url[:50]}, "K"
            )
            # #endregion
            return False

        # Block javascript: and data: URLs
        if url.startswith(("javascript:", "data:", "file:")):
            # #region debug instrumentation
            _debug_log(
                "guardrails.py:151", "_is_safe_url failed - blocked scheme", {"url": url[:50]}, "K"
            )
            # #endregion
            return False

        # Block localhost/private IPs in production (optional)
        # if re.search(r'localhost|127\.0\.0\.1|192\.168\.|10\.', url):
        #     return False

        # #region debug instrumentation
        _debug_log("guardrails.py:159", "_is_safe_url passed", {"url": url[:50]}, "K")
        # #endregion
        return True

    def sanitize_response(self, response: str) -> str:
        """Sanitize agent response to remove potentially harmful content."""
        if not isinstance(response, str):
            return ""

        # Remove script tags
        response = re.sub(
            r"<script[^>]*>.*?</script>", "", response, flags=re.IGNORECASE | re.DOTALL
        )

        # Remove javascript: URLs
        response = re.sub(r'javascript:[^"\s]*', "", response, flags=re.IGNORECASE)
        # Remove data: URLs (common XSS vector)
        response = re.sub(r'data:[^"\s]*', "", response, flags=re.IGNORECASE)

        # Replace known placeholder link patterns to avoid misleading users
        # (We prefer an explicit "not available" over placeholder hallucinations)
        placeholder_patterns = [
            r"\[(?:CVS|Walmart|Walgreens|Target|Amazon|Store|Website|Product|Online)\s+(?:Website|Store|Link|Purchase|Buy)\s*Link?\]",
            r"\[(?:Website|Store|Product|Online|Purchase|Buy)\s*Link?\]",
            r"\[Link\s*to\s*[^\]]+\]",
            r"\[[^\]]*Link[^\]]*\]",
            r"\[(?:product_url|Product URL|PRODUCT_URL)\]",
        ]
        for pat in placeholder_patterns:
            response = re.sub(pat, "Product link not available", response, flags=re.IGNORECASE)

        # Remove excessive whitespace
        response = re.sub(r"\s{3,}", " ", response)

        return response.strip()

    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        """Extract domain from URL."""
        if not url or not isinstance(url, str):
            return None
        if not url.startswith(("http://", "https://")):
            return None
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return None

    def _collect_allowed_domains(
        self, products: List[Dict[str, Any]], merchant_domains: Dict[str, List[str]]
    ) -> tuple[set[str], Dict[str, bool]]:
        """Collect allowed domains from products and check which merchants are present."""
        allowed_domains: set[str] = set()
        present: Dict[str, bool] = {m: False for m in merchant_domains.keys()}
        for p in products or []:
            url = p.get("product_url") or p.get("link") or p.get("url")
            d = self._extract_domain(url) if isinstance(url, str) else None
            if not d:
                continue
            allowed_domains.add(d)
            for merchant, domains in merchant_domains.items():
                if any(d == dom or d.endswith("." + dom) or dom in d for dom in domains):
                    present[merchant] = True
        return allowed_domains, present

    def _find_mentioned_merchants(
        self, response: str, merchant_domains: Dict[str, List[str]]
    ) -> List[str]:
        """Find which merchants are mentioned in the response text."""
        text_lower = response.lower()
        return [m for m in merchant_domains.keys() if re.search(rf"\b{re.escape(m)}\b", text_lower)]

    def _strip_untrusted_markdown_links(self, text: str, allowed_domains: set[str]) -> str:
        """Strip markdown links to domains not in allowed_domains."""

        def repl(m):
            url = (m.group(2) or "").strip()
            d = self._extract_domain(url)
            if not d or (allowed_domains and d not in allowed_domains):
                return "View Product"
            return m.group(0)

        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, text)

    def _neutralize_merchant_mentions(self, response: str, bad_merchants: List[str]) -> str:
        """Neutralize mentions of merchants that aren't present in product URLs."""
        for m in bad_merchants:
            response = re.sub(
                rf"(?i)\b(buy|purchase|order)\s+(on|from)\s+{re.escape(m)}\b", "Buy here", response
            )
            response = re.sub(
                rf"(?i)\b(view|check|shop)\s+(on|at|from)\s+{re.escape(m)}\b",
                "View Product",
                response,
            )
            response = re.sub(
                rf"(?i)\bavailable\s+(on|at)\s+{re.escape(m)}\b", "Available online", response
            )
            response = re.sub(
                rf"(?i)\b(where to buy|buy at|shop at)\s*:\s*{re.escape(m)}\b", "Buy here", response
            )
            response = re.sub(rf"(?i)\b(on|at|from|via)\s+{re.escape(m)}\b", "online", response)
            response = re.sub(rf"(?i)(,|\band)\s+{re.escape(m)}\b", "", response)
        return response

    def _clean_lines_with_merchants(self, response: str, bad_merchants: List[str]) -> str:
        """Remove merchant mentions from lines containing buy/purchase/order keywords."""
        lines = response.splitlines()
        cleaned_lines: List[str] = []
        for line in lines:
            line_lower = line.lower()
            if "buy" in line_lower or "purchase" in line_lower or "order" in line_lower:
                for m in bad_merchants:
                    line = re.sub(rf"(?i)\b{re.escape(m)}\b", "", line).strip()
                    line = re.sub(r"\s{2,}", " ", line)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def sanitize_response_with_products(self, response: str, products: List[Dict[str, Any]]) -> str:
        """Sanitize response using product context to prevent merchant/store hallucinations.

        If the response claims a merchant (e.g., "Buy on Amazon") but none of the returned
        product URLs are actually from that merchant domain, we remove/neutralize that claim.
        """
        response = self.sanitize_response(response)
        if not response or not products:
            return response

        merchant_domains = {
            "amazon": ["amazon.com", "amazon.in", "amazon.co.uk", "amzn.to"],
            "walmart": ["walmart.com"],
            "target": ["target.com"],
            "ebay": ["ebay.com", "ebay.co.uk"],
            "best buy": ["bestbuy.com"],
            "costco": ["costco.com"],
            "sephora": ["sephora.com"],
        }

        allowed_domains, present = self._collect_allowed_domains(products, merchant_domains)
        mentioned = self._find_mentioned_merchants(response, merchant_domains)
        bad = [m for m in mentioned if not present.get(m, False)]

        response = self._strip_untrusted_markdown_links(response, allowed_domains)

        if not bad:
            return response

        logger.warning(
            f"Merchant hallucination detected in response: {bad}. Neutralizing store mentions."
        )

        response = self._neutralize_merchant_mentions(response, bad)
        response = self._clean_lines_with_merchants(response, bad)

        return response

    def check_content_safety(self, content: str) -> Tuple[bool, Optional[str], str]:
        """
        Check content for safety issues.

        Returns:
            (is_safe, violation_type, severity)
        """
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(content):
                return False, "blocked_content", "high"

        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                return True, "suspicious_content", "low"  # Warn but allow

        return True, None, "none"

    def validate_session_id(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """Validate session ID format."""
        if not session_id or not isinstance(session_id, str):
            return False, "Session ID must be a non-empty string"

        if len(session_id) > 100:
            return False, "Session ID too long"

        # Check for suspicious characters
        if re.search(r'[<>"\']', session_id):
            return False, "Session ID contains invalid characters"

        return True, None


# Global guardrails instance
guardrails = ShoppingAssistantGuardrails()
