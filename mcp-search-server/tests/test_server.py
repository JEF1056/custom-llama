"""Tests for the MCP Search Server."""

import pytest


def test_import():
    """Test that the server can be imported."""
    from src.server import create_server

    server = create_server()
    assert server is not None


def test_config():
    """Test that the config can be imported."""
    from src.config import settings

    assert settings.MCP_SERVER_HOST == "0.0.0.0"
    assert settings.MCP_SERVER_PORT == 3100
    assert settings.SEARCH_ENGINE == "duckduckgo"
    assert settings.MAX_RESULTS == 10


def test_search_models():
    """Test the search models."""
    from src.search.models import SearchResult, SearchResponse

    result = SearchResult(
        title="Test",
        url="https://example.com",
        snippet="Test snippet",
    )
    assert result.title == "Test"
    assert result.url == "https://example.com"
    assert result.snippet == "Test snippet"

    response = SearchResponse(query="test", results=[result], total=1)
    assert response.query == "test"
    assert len(response.results) == 1
    assert response.total == 1
