"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_search_products():
    """Test product search endpoint."""
    response = client.get("/api/products/search?q=headphones")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data


@pytest.mark.asyncio
async def test_chat_endpoint():
    """Test chat endpoint."""
    response = client.post(
        "/api/chat/",
        json={"message": "Find me headphones"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data

