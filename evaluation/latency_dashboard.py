"""Latency monitoring and reporting."""
from src.analytics.latency_tracker import latency_tracker
import json
from datetime import datetime
from typing import Dict, Any, List


class LatencyDashboard:
    """Generate latency reports and identify bottlenecks."""
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive latency report."""
        stats = latency_tracker.get_all_stats()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "components": stats,
            "summary": self._generate_summary(stats),
            "bottlenecks": self._identify_bottlenecks(stats),
            "recommendations": self._generate_recommendations(stats)
        }
        
        return report
    
    def _generate_summary(self, stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not stats:
            return {"message": "No latency data available"}
        
        total_avg = sum(s.get("mean", 0) for s in stats.values())
        
        # Find slowest and fastest components
        component_means = [(name, s.get("mean", 0)) for name, s in stats.items()]
        slowest = max(component_means, key=lambda x: x[1]) if component_means else ("N/A", 0)
        fastest = min(component_means, key=lambda x: x[1]) if component_means else ("N/A", 0)
        
        return {
            "total_average_latency": total_avg,
            "slowest_component": {"name": slowest[0], "latency": slowest[1]},
            "fastest_component": {"name": fastest[0], "latency": fastest[1]},
            "p95_total": sum(s.get("p95", 0) for s in stats.values()),
            "p99_total": sum(s.get("p99", 0) for s in stats.values()),
            "total_requests": max((s.get("count", 0) for s in stats.values()), default=0)
        }
    
    def _identify_bottlenecks(self, stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify latency bottlenecks."""
        bottlenecks = []
        
        for component, stat in stats.items():
            mean = stat.get("mean", 0)
            p95 = stat.get("p95", 0)
            
            if mean > 0.5:  # More than 500ms
                bottlenecks.append({
                    "component": component,
                    "average_latency": mean,
                    "p95_latency": p95,
                    "severity": "high" if mean > 2.0 else "medium" if mean > 1.0 else "low"
                })
        
        return sorted(bottlenecks, key=lambda x: x["average_latency"], reverse=True)
    
    def _generate_recommendations(self, stats: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        api_latency = stats.get("serper_api_call", {}).get("mean", 0)
        if api_latency > 1.0:
            recommendations.append(
                f"Serper API calls average {api_latency:.2f}s - Consider caching API responses"
            )
        
        rag_latency = stats.get("rag_retrieval", {}).get("mean", 0)
        if rag_latency > 0.5:
            recommendations.append(
                f"RAG retrieval averages {rag_latency:.2f}s - Optimize by reducing candidates or caching embeddings"
            )
        
        llm_latency = stats.get("llm_processing", {}).get("mean", 0)
        if llm_latency > 2.0:
            recommendations.append(
                f"LLM processing averages {llm_latency:.2f}s - Consider reducing context size or using streaming"
            )
        
        embedding_latency = stats.get("embedding_generation", {}).get("mean", 0)
        if embedding_latency > 0.1:
            recommendations.append(
                f"Embedding generation averages {embedding_latency:.2f}s - Cache embeddings for common queries"
            )
        
        if not recommendations:
            recommendations.append("Latency looks good! No major bottlenecks identified.")
        
        return recommendations
    
    def print_report(self):
        """Print formatted latency report."""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("LATENCY PERFORMANCE REPORT")
        print("="*60)
        print(f"Generated: {report['timestamp']}")
        
        print("\n--- Summary ---")
        summary = report['summary']
        print(f"Total Average Latency: {summary.get('total_average_latency', 0):.3f}s")
        print(f"P95 Total Latency: {summary.get('p95_total', 0):.3f}s")
        print(f"P99 Total Latency: {summary.get('p99_total', 0):.3f}s")
        
        # TTFT stats if available
        ttft_stats = report['components'].get('ttft')
        if ttft_stats:
            print(f"TTFT Average: {ttft_stats.get('mean', 0):.3f}s")
            print(f"TTFT P95: {ttft_stats.get('p95', 0):.3f}s")
            print(f"TTFT P99: {ttft_stats.get('p99', 0):.3f}s")
        
        print("\n--- Component Breakdown ---")
        for component, stats in report['components'].items():
            print(f"\n{component}:")
            print(f"  Count: {stats.get('count', 0)}")
            print(f"  Mean: {stats.get('mean', 0):.3f}s")
            print(f"  P50: {stats.get('p50', 0):.3f}s")
            print(f"  P95: {stats.get('p95', 0):.3f}s")
            print(f"  P99: {stats.get('p99', 0):.3f}s")
        
        print("\n--- Bottlenecks ---")
        bottlenecks = report['bottlenecks']
        if bottlenecks:
            for bottleneck in bottlenecks:
                severity_icon = "🔴" if bottleneck['severity'] == 'high' else "🟡" if bottleneck['severity'] == 'medium' else "🟢"
                print(f"{severity_icon} {bottleneck['component']}: {bottleneck['average_latency']:.3f}s (P95: {bottleneck['p95_latency']:.3f}s)")
        else:
            print("No significant bottlenecks found.")
        
        print("\n--- Recommendations ---")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)


# Usage example
if __name__ == "__main__":
    dashboard = LatencyDashboard()
    dashboard.print_report()

