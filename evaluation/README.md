# Comprehensive Evaluation Framework

Complete evaluation framework with DeepEval, Langfuse, IR metrics, and performance tracking for the shopping assistant.

## 📊 Available Metrics

### LLM Quality Metrics (DeepEval)
1. **Groundedness** - Verify responses are grounded in retrieved products
2. **Answer Relevancy** - Measure how relevant responses are to queries
3. **Contextual Relevancy** - Evaluate if retrieved context is relevant
4. **Contextual Precision** - Measure precision of context retrieval
5. **Summarization** - Evaluate response completeness
6. **Bias** - Check for bias in responses
7. **Tone** - Evaluate response tone appropriateness
8. **Coherence** - Measure response coherence

### Retrieval Quality Metrics (IR Metrics)
1. **NDCG@K** - Normalized Discounted Cumulative Gain (ranking quality)
2. **Recall@K** - Fraction of relevant items found
3. **Precision@K** - Fraction of top K that are relevant
4. **Context Precision** - Precision of items in LLM context
5. **MRR** - Mean Reciprocal Rank
6. **MAP** - Mean Average Precision

### Performance Metrics
1. **Total Time** - End-to-end response time
2. **TTFT** - Time To First Token
3. **Component Breakdown** - Latency per component
4. **Throughput** - Queries per second

## 🚀 Quick Start

### Option 1: CLI Interface (Recommended)

```bash
# Run all evaluations
python -m evaluation.cli --type all --queries evaluation/datasets/test_queries.json

# Run specific evaluation type
python -m evaluation.cli --type llm --queries evaluation/datasets/test_queries.json
python -m evaluation.cli --type performance --iterations 5
python -m evaluation.cli --type retrieval --dataset evaluation/datasets/relevance_labels.json
```

### Option 2: Python API

```python
from evaluation.unified_evaluator import UnifiedEvaluator

evaluator = UnifiedEvaluator()
results = await evaluator.evaluate_batch(
    queries=["Find me wireless headphones under $100"],
    evaluation_types=["full_system"]
)
```

### 1. Create Relevance Dataset

```bash
python evaluation/create_relevance_dataset.py
```

This creates `evaluation/relevance_dataset.json` with:
- All queries from recent evaluation
- Retrieved products for each query
- Empty `relevant_products` dict (to be filled)

### 2. Label Products

Open `evaluation/relevance_dataset.json` and for each query:

1. Review `retrieved_products`
2. Label with relevance scores (0-4):
   - **0** = Not relevant
   - **1** = Somewhat relevant
   - **2** = Relevant
   - **3** = Highly relevant
   - **4** = Perfect match
3. Update `relevant_products`:
   ```json
   "relevant_products": {
     "prod_prod_001": 4,
     "prod_prod_007": 1
   }
   ```

### 3. Run Evaluation

```bash
python evaluation/evaluate_ir_metrics.py
```

Output includes:
- Per-query metrics
- Aggregate metrics
- Performance assessment
- Detailed JSON report

## 📁 Files

### Core Evaluation
- `unified_evaluator.py` - Main orchestrator combining all metrics
- `cli.py` - Command-line interface
- `report_generator.py` - Generate reports (JSON, CSV, HTML)

### DeepEval Integration
- `deepeval_config.py` - DeepEval configuration
- `deepeval_integration.py` - DeepEval test cases and metrics

### Langfuse Integration
- `langfuse_integration.py` - Langfuse evaluation export
- `src/analytics/langfuse_client.py` - Langfuse client and tracing

### Test Suites
- `test_suites/llm_quality_tests.py` - LLM quality tests
- `test_suites/retrieval_tests.py` - Retrieval quality tests
- `test_suites/performance_tests.py` - Performance tests
- `test_suites/system_tests.py` - Full system integration tests

### IR Metrics (Legacy)
- `ir_metrics.py` - All IR metrics implementations
- `evaluate_ir_metrics.py` - IR metrics evaluation script
- `create_relevance_dataset.py` - Dataset creation helper
- `relevance_dataset.json` - Your labeled dataset

### Datasets
- `datasets/test_queries.json` - Standard test queries
- `datasets/relevance_labels.json` - Ground truth labels

## 🎯 Target Goals

### LLM Quality (DeepEval)
| Metric | Target | Status |
|--------|--------|--------|
| Groundedness | ≥ 0.70 | ⏳ To measure |
| Answer Relevancy | ≥ 0.70 | ⏳ To measure |
| Contextual Relevancy | ≥ 0.70 | ⏳ To measure |
| Summarization | ≥ 0.70 | ⏳ To measure |
| Bias | ≥ 0.80 | ⏳ To measure |
| Tone | ≥ 0.70 | ⏳ To measure |
| Coherence | ≥ 0.70 | ⏳ To measure |

### Retrieval Quality (IR Metrics)
| Metric | Target | Status |
|--------|--------|--------|
| Precision@5 | ≥ 0.65 | ⏳ To measure |
| Recall@10 | ≥ 0.60 | ⏳ To measure |
| NDCG@10 | ≥ 0.75 | ⏳ To measure |
| Context Precision | ≥ 0.60 | ⏳ To measure |
| MRR | ≥ 0.70 | ⏳ To measure |

### Performance
| Metric | Target | Status |
|--------|--------|--------|
| Average Total Time | < 5.0s | ⏳ To measure |
| Average TTFT | < 2.0s | ⏳ To measure |
| Success Rate | ≥ 95% | ⏳ To measure |

## 🔧 Configuration

### Environment Variables

Add to `.env`:
```env
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROJECT_NAME=shopping-assistant
LANGFUSE_ENABLED=true

# DeepEval Configuration
DEEPEVAL_API_KEY=your_api_key
DEEPEVAL_ENABLED=true
```

### Langfuse Integration

All queries are automatically traced to Langfuse when enabled. Evaluation results are exported to Langfuse for:
- Historical tracking
- Score comparison over time
- Trace-based evaluation
- Cost tracking
- Performance analytics

## 📖 Documentation

See individual test suite files for:
- Detailed metric explanations
- Interpretation guidelines
- Troubleshooting tips
- Best practices


