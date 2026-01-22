"""LLM response quality metrics evaluation."""
import re
from typing import List, Dict, Any, Set


class LLMQualityMetrics:
    """Evaluate LLM response quality."""
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        products: List[Dict[str, Any]] = None,
        tools_used: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate response quality across multiple dimensions.
        
        Returns:
            Dict with quality scores (0-1):
            - overall_score: Weighted average
            - relevance_score: How relevant is the response to query
            - completeness_score: How complete is the response
            - tool_usage_score: Appropriate tool usage
            - structure_score: Response structure and formatting
        """
        if products is None:
            products = []
        if tools_used is None:
            tools_used = []
        
        # Calculate individual scores
        relevance = self._calculate_relevance(query, response, products)
        completeness = self._calculate_completeness(query, response, products)
        tool_usage = self._calculate_tool_usage(query, response, tools_used, products)
        structure = self._calculate_structure(response)
        
        # Weighted overall score
        overall = (
            relevance * 0.35 +
            completeness * 0.30 +
            tool_usage * 0.25 +
            structure * 0.10
        )
        
        return {
            "overall_score": round(overall, 3),
            "relevance_score": round(relevance, 3),
            "completeness_score": round(completeness, 3),
            "tool_usage_score": round(tool_usage, 3),
            "structure_score": round(structure, 3)
        }
    
    def _calculate_relevance(
        self,
        query: str,
        response: str,
        products: List[Dict[str, Any]]
    ) -> float:
        """Calculate how relevant the response is to the query."""
        score = 0.0
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'me', 'find', 'show', 'tell', 'what', 'where', 'when', 'how', 'why'}
        query_terms = {t for t in query_terms if t not in stop_words and len(t) > 2}
        
        if not query_terms:
            return 0.5  # Default if no meaningful terms
        
        # Check if response mentions query terms
        mentioned_terms = sum(1 for term in query_terms if term in response_lower)
        term_coverage = mentioned_terms / len(query_terms) if query_terms else 0
        score += term_coverage * 0.4
        
        # Check if products match query intent
        if products:
            # Check if product names/descriptions relate to query
            matching_products = 0
            for product in products:
                product_text = (
                    product.get('name', '') + ' ' + 
                    product.get('description', '') + ' ' +
                    product.get('category', '')
                ).lower()
                
                # Check if any query term appears in product
                if any(term in product_text for term in query_terms):
                    matching_products += 1
            
            product_relevance = matching_products / len(products) if products else 0
            score += product_relevance * 0.4
        
        # Check if response directly addresses query type
        question_words = {'what', 'where', 'when', 'who', 'why', 'how', 'which'}
        if any(qw in query_lower for qw in question_words):
            # For questions, check if response provides answer
            if len(response) > 50:  # Substantive response
                score += 0.2
        else:
            # For commands/requests, check if response acknowledges
            if any(word in response_lower for word in ['here', 'found', 'showing', 'showing', 'results']):
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_completeness(
        self,
        query: str,
        response: str,
        products: List[Dict[str, Any]]
    ) -> float:
        """Calculate how complete the response is."""
        score = 0.0
        
        # Response length (longer responses often more complete)
        if len(response) < 50:
            score += 0.2
        elif len(response) < 200:
            score += 0.4
        else:
            score += 0.5
        
        # Check if response includes product information
        if products:
            # Response should mention products
            product_mentions = sum(1 for p in products if p.get('name', '').lower() in response.lower())
            if product_mentions > 0:
                score += 0.3
            
            # Check if response includes key product details
            detail_indicators = ['price', 'rating', 'description', 'available', 'stock']
            details_mentioned = sum(1 for indicator in detail_indicators if indicator in response.lower())
            score += min(details_mentioned / len(detail_indicators), 0.2)
        
        # Check for structured information (lists, numbers, etc.)
        if re.search(r'\d+', response):  # Contains numbers
            score += 0.1
        if re.search(r'[•\-\*]', response) or '\n' in response:  # Contains lists
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_tool_usage(
        self,
        query: str,
        response: str,
        tools_used: List[str],
        products: List[Dict[str, Any]]
    ) -> float:
        """Calculate if tools were used appropriately."""
        score = 0.0
        
        # Check if query requires product search
        search_indicators = ['find', 'search', 'show', 'get', 'buy', 'purchase', 'product', 'item']
        requires_search = any(indicator in query.lower() for indicator in search_indicators)
        
        if requires_search:
            # Should have used search_products tool
            if 'search_products' in tools_used:
                score += 0.5
            elif any('search' in tool for tool in tools_used):
                score += 0.3
            else:
                # No search tool used but products found - might be from context
                if products:
                    score += 0.2
                else:
                    score += 0.1  # Should have searched but didn't
        
        # Check if products were found
        if products:
            score += 0.3
            # More products = better (up to a point)
            if len(products) >= 5:
                score += 0.2
            elif len(products) >= 1:
                score += 0.1
        
        # Check if appropriate tools were used for query type
        if 'price' in query.lower() and any('price' in tool for tool in tools_used):
            score += 0.1
        if 'available' in query.lower() or 'stock' in query.lower():
            if any('availability' in tool for tool in tools_used):
                score += 0.1
        
        # Penalize if too many tools used unnecessarily
        if len(tools_used) > 3:
            score *= 0.9  # Slight penalty
        
        return min(score, 1.0)
    
    def _calculate_structure(self, response: str) -> float:
        """Calculate response structure and formatting quality."""
        score = 0.0
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) > 1:
            score += 0.3
        
        # Check for capitalization
        if response and response[0].isupper():
            score += 0.2
        
        # Check for proper punctuation
        if response.rstrip().endswith(('.', '!', '?')):
            score += 0.2
        
        # Check for formatting (lists, paragraphs)
        if '\n' in response or '\n\n' in response:
            score += 0.2
        
        # Check for reasonable length (not too short, not too long)
        if 50 <= len(response) <= 2000:
            score += 0.1
        
        return min(score, 1.0)


