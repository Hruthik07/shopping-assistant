# Setup .env file for E-Commerce Shopping Assistant
Write-Host "=== Environment Setup ===" -ForegroundColor Green
Write-Host ""

# Check if .env already exists
if (Test-Path .env) {
    Write-Host "⚠️  .env file already exists!" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to overwrite it? (y/n)"
    if ($overwrite -ne "y") {
        Write-Host "Cancelled." -ForegroundColor Red
        exit
    }
}

# Get API key from user
Write-Host "Enter your Anthropic API Key:" -ForegroundColor Cyan
Write-Host "(You can get it from: https://console.anthropic.com/)" -ForegroundColor Gray
$apiKey = Read-Host "ANTHROPIC_API_KEY"

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host "❌ API key cannot be empty!" -ForegroundColor Red
    exit 1
}

# Create .env file content
$envContent = @"
# LLM Configuration
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-20241022
LLM_TEMPERATURE=0.3

# Anthropic API Key
ANTHROPIC_API_KEY=$apiKey

# Optional: Other API Keys for enhanced features
# SERPER_API_KEY=your-serper-key-here
# TAVILY_API_KEY=your-tavily-key-here

# Database (SQLite by default)
DATABASE_URL=sqlite:///./shopping_assistant.db

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
USE_OPENAI_EMBEDDINGS=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=3565
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"@

# Write to .env file
$envContent | Out-File -FilePath .env -Encoding utf8 -NoNewline

Write-Host ""
Write-Host "✅ .env file created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now restart your server to use the API key." -ForegroundColor Cyan





