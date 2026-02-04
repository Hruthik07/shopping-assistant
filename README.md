# 🛍️ AI Shopping Assistant

![CI](https://github.com/Hruthik07/shopping-assistant/workflows/CI/badge.svg)
![Code Quality](https://github.com/Hruthik07/shopping-assistant/workflows/Code%20Quality/badge.svg)
![Security Scan](https://github.com/Hruthik07/shopping-assistant/workflows/Security%20Scan/badge.svg)
![Build Verification](https://github.com/Hruthik07/shopping-assistant/workflows/Build%20Verification/badge.svg)

> An intelligent shopping companion that understands what you're looking for, finds the best deals across multiple retailers, and helps you make informed purchasing decisions—all through natural conversation.

---

## What Problem Does This Solve?

Shopping online can be overwhelming. You're faced with:
- **Too many options** across dozens of websites
- **Price hunting** that requires manually checking multiple retailers
- **Information overload** from conflicting reviews and specifications
- **Hidden deals** that you might miss
- **Time-consuming research** to find the right product within your budget

This AI Shopping Assistant solves all of these problems by acting as your personal shopping expert. It:
- **Searches multiple retailers simultaneously** (Amazon, eBay, Walmart, Best Buy, and more)
- **Compares prices automatically** and highlights the best deals
- **Understands natural language**—just tell it what you need in plain English
- **Remembers your preferences** and previous conversations
- **Detects price drops and deals** automatically
- **Provides personalized recommendations** based on your budget and requirements
- **Speaks to you** with voice input and output support

Think of it as having a knowledgeable shopping assistant who never sleeps, has access to every online store, and always has your best interests in mind.

---

## 🎯 Core Features

### Intelligent Product Discovery
- **Multi-Source Search**: Queries Google Shopping, Amazon, eBay, Walmart, and Best Buy in parallel
- **RAG-Powered Retrieval**: Uses advanced Retrieval-Augmented Generation to find relevant products from a knowledge base of reviews, FAQs, and product descriptions
- **Hybrid Search**: Combines semantic understanding with keyword matching for precise results
- **Real-Time Data**: Always shows current prices, availability, and product information

### Smart Deal Finding
- **Automatic Price Comparison**: Compares prices across retailers instantly
- **Deal Detection**: Identifies price drops, seasonal sales, and limited-time offers
- **Price History Tracking**: Monitors price changes over time to catch the best deals
- **Deal Badges**: Visual indicators for "Best Price", "Save X%", "Limited Time", and more
- **Customer-First Ranking**: Transparent algorithm that prioritizes the best deals—no affiliate bias

### Conversational AI
- **Natural Language Understanding**: Ask questions like "Find me wireless headphones under $200" or "What's the best laptop for a student?"
- **Context Awareness**: Remembers your previous requests and preferences throughout the conversation
- **Multi-Step Reasoning**: Handles complex queries that require multiple steps (e.g., "Find a laptop, then add it to my cart")
- **Personalized Responses**: Adapts its tone and recommendations based on your needs

### Voice Assistant
- **Speech-to-Text**: Speak your queries instead of typing
- **Text-to-Speech**: Hear responses read aloud
- **Customizable Voice**: Adjust speech rate, pitch, and volume to your preference
- **Browser-Based**: Works entirely in your browser—no additional software needed

### Modern User Interface
- **Beautiful Design**: Glassmorphism effects, smooth animations, and modern visual aesthetics
- **Interactive Backgrounds**: Dynamic particle effects and gradient animations that respond to your cursor
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Accessibility**: Keyboard navigation, screen reader support, and reduced motion options
- **Real-Time Updates**: Live product results as you chat

### Production-Ready Infrastructure
- **FastAPI Backend**: High-performance REST API with WebSocket support for streaming responses
- **Redis Caching**: Intelligent caching reduces latency and API costs
- **Database Persistence**: SQLAlchemy with support for SQLite and PostgreSQL
- **Vector Database**: ChromaDB for semantic search and RAG capabilities
- **Docker Deployment**: Containerized for easy deployment anywhere
- **CI/CD Pipeline**: Automated testing, code quality checks, and security scanning
- **Comprehensive Monitoring**: Health checks, metrics, error tracking, and performance analytics

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.11+** - Latest Python features and performance improvements
- **LangChain** - Framework for building LLM-powered applications
- **LangGraph** - For complex agent workflows and multi-step reasoning
- **OpenAI / Anthropic** - LLM providers (GPT-4, Claude 3.5)
- **ChromaDB** - Vector database for semantic search
- **SQLAlchemy** - Database ORM with migration support
- **Redis** - High-performance caching layer
- **WebSockets** - Real-time bidirectional communication

### AI & Machine Learning
- **RAG (Retrieval-Augmented Generation)** - Combines retrieval with generation for accurate responses
- **Sentence Transformers** - For generating embeddings and semantic search
- **Hybrid Search** - Combines vector similarity with keyword matching
- **Model Context Protocol (MCP)** - Standardized tool integration

### Frontend
- **Vanilla JavaScript** - No framework dependencies, pure performance
- **Modern CSS** - Custom properties, animations, glassmorphism effects
- **Web Speech API** - Browser-native speech recognition and synthesis
- **Responsive Design** - Mobile-first approach with flexible layouts

### DevOps & Monitoring
- **Docker** - Containerization for consistent deployments
- **GitHub Actions** - CI/CD pipeline with automated testing
- **Pytest** - Comprehensive test suite with coverage reporting
- **Black** - Code formatting
- **Flake8** - Linting
- **MyPy** - Type checking
- **pip-audit** - Security vulnerability scanning
- **Langfuse** - LLM observability and tracing
- **CloudWatch** - Metrics and monitoring (for AWS deployments)

### External APIs & Services
- **Serper API** - Google Shopping integration
- **Tavily** - Web search and information retrieval
- **Amazon Product Advertising API** - Amazon product data
- **eBay Finding API** - eBay product listings
- **Walmart Open API** - Walmart product catalog
- **Best Buy API** - Best Buy product information

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Redis (optional, for caching)
- API keys:
  - OpenAI API key OR Anthropic API key (for the LLM)
  - Serper API key (for Google Shopping) - [Get it here](https://serper.dev)
  - Tavily API key (optional, for web search) - [Get it here](https://tavily.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hruthik07/shopping-assistant.git
   cd shopping-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   # Required
   OPENAI_API_KEY=your_openai_key
   # OR
   ANTHROPIC_API_KEY=your_anthropic_key
   
   # Required for product search
   SERPER_API_KEY=your_serper_key
   
   # Optional
   TAVILY_API_KEY=your_tavily_key
   DATABASE_URL=sqlite:///./shopping_assistant.db
   CACHE_ENABLED=true
   REDIS_URL=redis://localhost:6379/0
   ```

5. **Initialize the database**
   ```bash
   python scripts/init_db.py
   ```

6. **Start the server**
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 3565 --reload
   ```

7. **Open your browser**
   
   Navigate to `http://localhost:3565` to access the web interface.

### Docker Deployment

For production deployment, use Docker Compose:

```bash
docker-compose up --build
```

The application will be available at `http://localhost:3565`.

---

## 📖 Usage Examples

### Web Interface

Simply open the application in your browser and start chatting:

- "Find me running shoes under $100"
- "What's the best wireless mouse for gaming?"
- "Show me laptops with at least 16GB RAM"
- "Compare prices for iPhone 15 across different stores"

### API Usage

#### Chat with the Assistant
```python
import requests

response = requests.post(
    "http://localhost:3565/api/chat/",
    json={"message": "Find me wireless headphones under $200"}
)

data = response.json()
print(data["response"])
print(f"Found {len(data.get('products', []))} products")
```

#### Search for Products
```bash
curl "http://localhost:3565/api/products/search?q=laptop&max_price=1000"
```

#### Get Conversation History
```bash
curl "http://localhost:3565/api/chat/history/{session_id}"
```

### WebSocket Streaming

For real-time streaming responses:

```javascript
const ws = new WebSocket('ws://localhost:3565/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.message);
};
```

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Agent Orchestrator               │
│  (LangChain + LangGraph)             │
└────────┬─────────────────────────────┘
         │
         ├──► RAG System (ChromaDB)
         │    └──► Hybrid Search (Semantic + Keyword)
         │
         ├──► MCP Tools
         │    ├──► Product Search
         │    ├──► Price Comparison
         │    ├──► Web Search
         │    └──► Cart Operations
         │
         ├──► Memory System
         │    ├──► Conversation History
         │    └──► User Preferences
         │
         └──► Multi-Step Reasoning
              └──► Tool Selection & Execution
                   │
                   ▼
         ┌─────────────────────┐
         │   Response +        │
         │   Product Results   │
         └─────────────────────┘
```

### Key Components

1. **Shopping Agent** (`src/agent/shopping_agent.py`)
   - Main orchestrator that processes queries
   - Integrates LLM, tools, and memory
   - Handles caching and error recovery

2. **RAG System** (`src/rag/`)
   - Vector store for semantic search
   - Document retrieval from product data
   - Hybrid search combining embeddings and keywords

3. **MCP Tools** (`src/mcp/tools/`)
   - Product search and aggregation
   - Price comparison and deal detection
   - Web search for additional information
   - Cart management operations

4. **Memory System** (`src/memory/`)
   - Conversation history tracking
   - User preference learning
   - Session management

5. **API Layer** (`src/api/`)
   - REST endpoints for chat and products
   - WebSocket support for streaming
   - Rate limiting and CORS handling

6. **Services** (`src/services/`)
   - Product aggregation from multiple sources
   - Price tracking and deal detection
   - Coupon matching and transparency

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

The project includes:
- Unit tests for core components
- Integration tests for API endpoints
- Test fixtures and mocks for external services
- Coverage reporting

---

## 📊 Monitoring & Observability

### Health Checks
- `GET /api/health` - Application health status
- `GET /api/metrics` - Performance metrics
- `GET /api/chat/cache/stats` - Cache statistics

### LLM Observability
- **Langfuse**: Traces all LLM calls, tracks token usage, and analyzes costs
- **CloudWatch**: Application metrics and Bedrock monitoring (for AWS deployments)

### Analytics
- Request latency tracking
- Cache hit/miss rates
- Error rate monitoring
- Deal detection analytics

---

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# LLM Configuration
LLM_PROVIDER=openai  # or anthropic
LLM_MODEL=gpt-4-turbo-preview  # or claude-3-5-haiku-20241022

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Database
DATABASE_URL=sqlite:///./shopping_assistant.db
# For PostgreSQL: postgresql://user:pass@localhost/dbname

# Caching
CACHE_ENABLED=true
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db

# Monitoring
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
```

---

## 📁 Project Structure

```
shopping-assistant/
├── src/
│   ├── agent/              # Agent orchestrator and workflows
│   │   ├── shopping_agent.py
│   │   ├── reasoning.py
│   │   └── prompts/
│   ├── api/                # FastAPI application
│   │   ├── main.py
│   │   ├── routes/         # API endpoints
│   │   └── websocket.py
│   ├── rag/                # RAG system
│   │   ├── retriever.py
│   │   ├── embeddings.py
│   │   └── document_loader.py
│   ├── mcp/                # MCP tools
│   │   ├── mcp_client.py
│   │   └── tools/          # Tool implementations
│   ├── memory/             # Conversation memory
│   │   ├── conversation_store.py
│   │   └── user_preferences.py
│   ├── services/           # Business logic
│   │   ├── product_aggregator.py
│   │   ├── deal_detector.py
│   │   └── price_comparison.py
│   ├── database/           # Database models
│   │   ├── models.py
│   │   └── schemas.py
│   └── utils/              # Utilities and config
├── frontend/               # Web interface
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/                  # Test suite
├── docker/                 # Docker configuration
├── .github/workflows/      # CI/CD pipelines
└── requirements.txt        # Python dependencies
```

---

## 🌟 What Makes This Special

This isn't just another chatbot. It's a **production-ready AI system** that:

1. **Actually finds products** - Not just text generation, but real product search across multiple retailers
2. **Saves you money** - Automatically compares prices and detects deals
3. **Remembers context** - Understands your preferences and previous conversations
4. **Works in real-time** - Live product data, not stale information
5. **Is transparent** - No hidden affiliate bias, just honest recommendations
6. **Is accessible** - Voice support, keyboard navigation, screen reader friendly
7. **Is beautiful** - Modern UI that's a joy to use
8. **Is reliable** - Comprehensive testing, monitoring, and error handling

---

## 🤝 Contributing

Contributions are welcome! Whether it's:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- UI/UX enhancements

Please feel free to open an issue or submit a pull request.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/) and [FastAPI](https://fastapi.tiangolo.com/)
- Product data from Google Shopping (via [Serper API](https://serper.dev))
- Inspired by real-world e-commerce AI assistants
- Thanks to the open-source community for amazing tools and libraries

---

## 📧 Support

For questions, issues, or feature requests, please open an issue on [GitHub](https://github.com/Hruthik07/shopping-assistant/issues).

---

**Built with ❤️ for smarter shopping**
