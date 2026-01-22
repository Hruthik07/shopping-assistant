# E-Commerce Shopping Assistant Agent

A production-ready AI-powered shopping assistant that combines **RAG (Retrieval-Augmented Generation)**, **MCP (Model Context Protocol)**, and modern AI engineering practices. This agent helps customers find products, answer questions, get recommendations, and complete purchases using real-time product data from e-commerce APIs.

## 🚀 Features

### Core Features
- **Real-Time Product Search**: Integrates with Google Shopping API (via Serper) for live product data
- **Multi-Source Aggregation**: Query multiple retailers (Amazon, eBay, Walmart, Best Buy) in parallel
- **RAG System**: Advanced retrieval with hybrid search (semantic + keyword) across products, reviews, and FAQs
- **MCP Tools**: 8+ tools including product search, price checking, web search, image analysis, and cart operations
- **Conversation Memory**: Tracks user preferences and conversation history
- **Multi-Step Workflows**: Intelligent agent workflows for complex shopping tasks
- **FastAPI Backend**: RESTful API with WebSocket support for streaming

### Deal Finding & Price Comparison
- **Price Comparison**: Compare prices across multiple retailers automatically
- **Deal Detection**: Identify price drops, best prices, and seasonal deals
- **Price History Tracking**: Track price changes over time to detect deals
- **Coupon Integration**: Match available coupons and promo codes to products
- **Customer-First Ranking**: Transparent ranking algorithm that prioritizes best deals (no affiliate bias)
- **Deal Badges**: Visual indicators for "Best Price", "Save X%", "Limited Time", etc.

### Production & Monitoring
- **Production Ready**: Docker deployment, rate limiting, logging, and analytics
- **Performance Optimized**: 30% latency reduction, parallelized operations, optimized context size
- **Comprehensive Monitoring**: Health checks, metrics endpoints, error tracking, cache statistics
- **Deal Analytics**: Track deal detection rates, savings, and price comparison effectiveness
- **Background Jobs**: Automated price tracking and data cleanup
- **Voice Assistant**: Browser-based speech-to-text and text-to-speech support

## 🏗️ Architecture

```
User Query → Agent Orchestrator → [RAG System | MCP Tools | Memory] → Multi-step Reasoning → Response
```

### Core Components

1. **RAG System**: Vector database (ChromaDB) with hybrid search
2. **MCP Integration**: Tool registry with 8+ tools
3. **Agent**: LangChain-based agent with tool calling
4. **Memory**: Conversation history and user preferences
5. **API**: FastAPI with REST and WebSocket endpoints

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key (for LLM) or Anthropic API key
- Serper API key (for Google Shopping) - [Get it here](https://serper.dev)
- Tavily API key (optional, for web search) - [Get it here](https://tavily.com)

### Optional: Additional Data Sources
- Amazon Product Advertising API (requires Associates account + 3 sales)
- eBay Finding API (free developer account)
- Walmart Open API (requires seller account)
- Best Buy API (free developer account)
- Coupon APIs: Honey, RetailMeNot (partner programs)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd agentic_ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required environment variables:
```bash
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key  # For Google Shopping
TAVILY_API_KEY=your_tavily_key  # Optional
```

5. **Initialize database**
```bash
python scripts/init_db.py
```

This creates all tables including the new `price_history` table for deal tracking.

6. **Initialize vector database** (happens automatically on first run)

## 🚀 Running the Application

### Development Mode

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Docker

```bash
docker-compose up --build
```

### Access API Documentation

- Swagger UI: `http://localhost:3565/docs`
- ReDoc: `http://localhost:3565/redoc`

### Health & Monitoring

#### Monitoring Stack

The application provides comprehensive monitoring through multiple tools:

1. **Langfuse** (LLM Observability)
   - Traces all LLM calls
   - Token usage tracking
   - Cost analysis
   - Response quality metrics
   - Configure: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

2. **CloudWatch** (AWS Bedrock Deployment)
   - Application metrics (latency, cache, errors)
   - Bedrock metrics (invocations, latency, errors)
   - Evaluation metrics (IR, DeepEval)
   - Cost tracking
   - Unified dashboard
   - Configure: `CLOUDWATCH_ENABLED=true`, `AWS_REGION`

3. **Confident AI** (Component Evaluation)
   - DeepEval test results
   - Quality score tracking
   - Automated evaluation pipeline

4. **IR Metrics** (Information Retrieval)
   - Precision@K, Recall@K, NDCG@K
   - Context Precision, MRR, MAP
   - Available via evaluation scripts

#### Health & Monitoring

- Health Check: `http://localhost:3565/api/health`
- Metrics: `http://localhost:3565/api/metrics`
- Cache Stats: `http://localhost:3565/api/chat/cache/stats`

## 📡 API Endpoints

### Chat Endpoints

- `POST /api/chat/` - Send a message to the shopping assistant
- `GET /api/chat/history/{session_id}` - Get conversation history

### Product Endpoints

- `GET /api/products/search?q={query}` - Search for products
- `GET /api/products/{product_id}` - Get product details

### Cart Endpoints

- `POST /api/cart/items` - Add item to cart
- `GET /api/cart/?user_id={id}` - Get shopping cart
- `DELETE /api/cart/items/{item_id}` - Remove item from cart

### WebSocket

- `WS /ws` - WebSocket endpoint for streaming chat

## 💡 Usage Examples

### Python Client

```python
import requests

# Chat with the assistant
response = requests.post(
    "http://localhost:8000/api/chat/",
    json={"message": "Find me wireless headphones under $200"}
)

data = response.json()
print(data["response"])
print(f"Found {len(data.get('products', []))} products")
```

### cURL

```bash
# Search for products
curl "http://localhost:8000/api/products/search?q=laptop"

# Chat with assistant
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the best running shoes?"}'
```

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/
```

## 🐳 Docker Deployment

Build and run with Docker Compose:

```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## 📁 Project Structure

```
agentic_ai/
├── src/
│   ├── agent/          # Agent orchestrator and workflows
│   ├── rag/            # RAG system (vector store, retriever)
│   ├── mcp/            # MCP tools (product, search, image, cart)
│   ├── memory/         # Conversation memory and preferences
│   ├── database/       # Database models and schemas
│   ├── api/            # FastAPI application and routes
│   ├── analytics/      # Logging and tracking
│   └── utils/          # Configuration and helpers
├── data/               # Sample product data
├── tests/              # Test files
├── docker/             # Docker configuration
└── requirements.txt   # Python dependencies
```

## 🔧 Configuration

Key configuration options in `.env`:

- `LLM_MODEL`: LLM model to use (default: gpt-4-turbo-preview)
- `EMBEDDING_MODEL`: Embedding model (default: sentence-transformers)
- `CHROMA_PERSIST_DIR`: Vector database directory
- `RATE_LIMIT_PER_MINUTE`: API rate limit

## 🌟 Key Technologies

- **LangChain/LangGraph**: Agent framework
- **ChromaDB**: Vector database
- **FastAPI**: Web framework
- **OpenAI**: LLM and embeddings
- **Serper API**: Google Shopping integration
- **Tavily**: Web search
- **SQLAlchemy**: Database ORM
- **Docker**: Containerization

## 📊 Features in Detail

### RAG System
- Multi-source retrieval (products, reviews, FAQs)
- Hybrid search (semantic + keyword)
- Real-time product fetching from APIs

### MCP Tools
1. **Product Tools**: Search, availability check, price check
2. **Search Tools**: Web search, product review search
3. **Image Tools**: Image analysis, similar product finder
4. **Cart Tools**: Add to cart, get cart, remove items

### Agent Capabilities
- Intent detection
- Tool selection
- Multi-step reasoning
- Context-aware responses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with LangChain and FastAPI
- Product data from Google Shopping (via Serper API)
- Inspired by real-world e-commerce AI assistants

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Make sure to add your API keys in the `.env` file before running the application. The system will fall back to local sample data if APIs are not configured.

