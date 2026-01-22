"""MCP (tool layer) evaluation suite.

This suite evaluates:
- Tool selection correctness (expected_tools vs tools_used)
- Constraint compliance as a proxy for tool-argument correctness (e.g., price range)
- Tool output validity (URLs, product fields)
- Response faithfulness to tool outputs (e.g., merchant mentions match URL domains)
- Reliability + latency over N iterations
"""

import asyncio
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from src.analytics.logger import logger


MERCHANT_DOMAINS = {
    "amazon": ["amazon.com", "amazon.in", "amazon.co.uk", "amzn.to"],
    "walmart": ["walmart.com"],
    "target": ["target.com"],
    "ebay": ["ebay.com", "ebay.co.uk"],
    "best buy": ["bestbuy.com"],
    "costco": ["costco.com"],
}


def _safe_float(price: Any) -> Optional[float]:
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        # Strip currency symbols and pick the first number
        m = re.search(r"(\d+(?:\.\d+)?)", price.replace(",", ""))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _extract_domain(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    if not url.startswith(("http://", "https://")):
        return None
    try:
        host = urlparse(url).netloc.lower()
        # strip leading www.
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None


def _mentions_merchant(response_text: str) -> List[str]:
    if not response_text:
        return []
    text = response_text.lower()
    mentioned = []
    for merchant in MERCHANT_DOMAINS.keys():
        # "target" is ambiguous (verb/noun). Only treat it as merchant when in shopping context.
        if merchant == "target":
            if re.search(r"\b(target\.com)\b", text):
                mentioned.append(merchant)
                continue
            if re.search(r"\b(buy|purchase|order|view|check|available|shop)\b.{0,20}\btarget\b", text):
                mentioned.append(merchant)
                continue
            if re.search(r"\b(on|at|from)\s+target\b", text):
                mentioned.append(merchant)
                continue
            continue
        # default: word-ish matching
        if re.search(rf"\b{re.escape(merchant)}\b", text):
            mentioned.append(merchant)
    return mentioned


def _merchant_domains_present(products: List[Dict[str, Any]]) -> Dict[str, bool]:
    present = {m: False for m in MERCHANT_DOMAINS.keys()}
    for p in products or []:
        url = p.get("product_url") or p.get("link") or p.get("url")
        domain = _extract_domain(url) if isinstance(url, str) else None
        if not domain:
            continue
        for merchant, domains in MERCHANT_DOMAINS.items():
            if any(domain == d or domain.endswith("." + d) or d in domain for d in domains):
                present[merchant] = True
    return present


def _check_price_constraints(
    products: List[Dict[str, Any]],
    min_price: Optional[float],
    max_price: Optional[float],
) -> Tuple[float, int, int]:
    """Return (compliance_rate, checked_count, compliant_count)."""
    checked = 0
    compliant = 0
    for p in products or []:
        price = _safe_float(p.get("price"))
        if price is None:
            continue
        checked += 1
        ok = True
        if min_price is not None and price < min_price:
            ok = False
        if max_price is not None and price > max_price:
            ok = False
        if ok:
            compliant += 1
    rate = (compliant / checked) if checked else 0.0
    return rate, checked, compliant


def _valid_url_rate(products: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    checked = 0
    valid = 0
    for p in products or []:
        url = p.get("product_url") or p.get("link") or p.get("url")
        if not url:
            continue
        checked += 1
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            valid += 1
    rate = (valid / checked) if checked else 0.0
    return rate, checked, valid


def _placeholder_present(response_text: str) -> bool:
    if not response_text:
        return False
    patterns = [
        r"\[product_url\]",
        r"\[Product URL\]",
        r"\[PRODUCT_URL\]",
        r"\[.*Website.*Link.*\]",
        r"\[.*Product.*Link.*\]",
    ]
    return any(re.search(p, response_text, re.IGNORECASE) for p in patterns)


@dataclass
class MCPQueryExpectation:
    query: str
    expected_tools: List[str]
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_products: int = 0
    allow_no_products: bool = False
    require_links: bool = True


class MCPTestSuite:
    """MCP evaluation suite."""

    def __init__(self, base_url: str = "http://localhost:3565"):
        self.base_url = base_url

    async def _call_chat(self, query: str, session_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(
                f"{self.base_url}/api/chat/",
                json={"message": query, "session_id": session_id},
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            return r.json()

    def _parse_expectation(self, item: Dict[str, Any]) -> MCPQueryExpectation:
        constraints = item.get("constraints") or {}
        return MCPQueryExpectation(
            query=item.get("query", ""),
            expected_tools=item.get("expected_tools", []) or [],
            min_price=constraints.get("min_price"),
            max_price=constraints.get("max_price"),
            min_products=constraints.get("min_products", 0),
            allow_no_products=bool(constraints.get("allow_no_products", False)),
            require_links=bool(constraints.get("require_links", True)),
        )

    async def evaluate_dataset(
        self,
        dataset: Dict[str, Any],
        iterations: int = 1,
    ) -> Dict[str, Any]:
        queries = dataset.get("queries", []) or []
        if not queries:
            raise ValueError("Dataset has no queries")

        all_results: List[Dict[str, Any]] = []

        for idx, item in enumerate(queries, 1):
            exp = self._parse_expectation(item)
            if not exp.query.strip():
                continue

            query_id = item.get("id") or f"mcp-{idx:03d}"
            per_iter: List[Dict[str, Any]] = []

            for it in range(iterations):
                session_id = f"mcp-eval-{query_id}-{it}"
                t0 = asyncio.get_event_loop().time()
                try:
                    api = await self._call_chat(exp.query, session_id=session_id)
                    elapsed = asyncio.get_event_loop().time() - t0

                    response_text = api.get("response", "") or ""
                    products = api.get("products", []) or []
                    tools_used = api.get("tools_used", []) or []

                    # Tool selection score
                    expected_set = set(exp.expected_tools)
                    used_set = set(tools_used)
                    missing_tools = list(expected_set - used_set)
                    tool_selection_pass = len(missing_tools) == 0
                    tool_selection_score = (
                        (len(expected_set & used_set) / len(expected_set)) if expected_set else 1.0
                    )

                    # Constraint compliance (proxy for argument correctness)
                    price_rate, price_checked, price_ok = _check_price_constraints(
                        products, exp.min_price, exp.max_price
                    )

                    # Product count expectation
                    product_count_pass = True
                    if not exp.allow_no_products:
                        product_count_pass = len(products) >= exp.min_products

                    # URL validity expectation
                    url_rate, url_checked, url_ok = _valid_url_rate(products)
                    url_pass = True
                    if exp.require_links:
                        # if we have products, we expect at least one valid URL
                        url_pass = (len(products) == 0 and exp.allow_no_products) or (url_ok >= 1)

                    # Response faithfulness: merchant mentions must match actual product URLs
                    mentioned = _mentions_merchant(response_text)
                    present = _merchant_domains_present(products)
                    bad_merchant_mentions = [m for m in mentioned if not present.get(m, False)]
                    faithfulness_pass = len(bad_merchant_mentions) == 0 and not _placeholder_present(response_text)

                    passed = (
                        tool_selection_pass
                        and product_count_pass
                        and url_pass
                        and faithfulness_pass
                        and (price_rate >= 0.8 if (exp.min_price is not None or exp.max_price is not None) else True)
                    )

                    per_iter.append(
                        {
                            "iteration": it + 1,
                            "success": True,
                            "elapsed": elapsed,
                            "tools_used": tools_used,
                            "products_count": len(products),
                            "tool_selection": {
                                "expected_tools": exp.expected_tools,
                                "missing_tools": missing_tools,
                                "score": round(tool_selection_score, 3),
                                "passed": tool_selection_pass,
                            },
                            "constraints": {
                                "min_price": exp.min_price,
                                "max_price": exp.max_price,
                                "price_checked": price_checked,
                                "price_compliant": price_ok,
                                "price_compliance_rate": round(price_rate, 3),
                                "min_products": exp.min_products,
                                "product_count_pass": product_count_pass,
                                "require_links": exp.require_links,
                                "url_checked": url_checked,
                                "url_valid": url_ok,
                                "url_valid_rate": round(url_rate, 3),
                                "url_pass": url_pass,
                            },
                            "faithfulness": {
                                "merchant_mentions": mentioned,
                                "bad_merchant_mentions": bad_merchant_mentions,
                                "placeholder_present": _placeholder_present(response_text),
                                "passed": faithfulness_pass,
                            },
                            "passed": passed,
                        }
                    )
                except Exception as e:
                    elapsed = asyncio.get_event_loop().time() - t0
                    logger.error(f"[MCP eval] Query failed ({query_id}) iter {it+1}: {e}")
                    per_iter.append(
                        {
                            "iteration": it + 1,
                            "success": False,
                            "elapsed": elapsed,
                            "error": str(e),
                            "passed": False,
                        }
                    )

            # Aggregate per query
            successes = [r for r in per_iter if r.get("success")]
            avg_latency = mean([r["elapsed"] for r in per_iter]) if per_iter else 0.0
            pass_rate = (sum(1 for r in per_iter if r.get("passed")) / len(per_iter)) if per_iter else 0.0

            # For scoring, average tool selection score across successful iters (or 0)
            avg_tool_score = (
                mean([r["tool_selection"]["score"] for r in successes if "tool_selection" in r]) if successes else 0.0
            )
            avg_price_rate = (
                mean([r["constraints"]["price_compliance_rate"] for r in successes if "constraints" in r])
                if successes
                else 0.0
            )
            avg_url_rate = (
                mean([r["constraints"]["url_valid_rate"] for r in successes if "constraints" in r]) if successes else 0.0
            )
            faithfulness_failures = sum(
                1 for r in successes if r.get("faithfulness", {}).get("passed") is False
            )

            all_results.append(
                {
                    "id": query_id,
                    "query": exp.query,
                    "iterations": iterations,
                    "avg_latency": round(avg_latency, 3),
                    "pass_rate": round(pass_rate, 3),
                    "avg_tool_selection_score": round(avg_tool_score, 3),
                    "avg_price_compliance_rate": round(avg_price_rate, 3),
                    "avg_url_valid_rate": round(avg_url_rate, 3),
                    "faithfulness_failures": faithfulness_failures,
                    "runs": per_iter,
                    "passed": pass_rate >= 1.0,  # strict: all iterations must pass
                }
            )

        # Aggregate suite
        total = len(all_results)
        passed = sum(1 for r in all_results if r.get("passed"))
        overall_pass_rate = (passed / total) if total else 0.0
        avg_latency_all = mean([r.get("avg_latency", 0.0) for r in all_results]) if all_results else 0.0
        avg_tool_all = mean([r.get("avg_tool_selection_score", 0.0) for r in all_results]) if all_results else 0.0
        avg_price_all = mean([r.get("avg_price_compliance_rate", 0.0) for r in all_results]) if all_results else 0.0
        avg_url_all = mean([r.get("avg_url_valid_rate", 0.0) for r in all_results]) if all_results else 0.0

        return {
            "total_queries": total,
            "successful": total,  # suite-level success is represented per query
            "failed": 0,
            "evaluation_types": ["mcp"],
            "aggregate_metrics": {
                "mcp": {
                    "pass_rate": round(overall_pass_rate, 3),
                    "avg_latency": round(avg_latency_all, 3),
                    "avg_tool_selection_score": round(avg_tool_all, 3),
                    "avg_price_compliance_rate": round(avg_price_all, 3),
                    "avg_url_valid_rate": round(avg_url_all, 3),
                }
            },
            "results": all_results,
        }

