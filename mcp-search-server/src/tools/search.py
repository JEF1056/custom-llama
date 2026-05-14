"""Search tool for MCP server."""

import json
import logging
from datetime import datetime

from mcp.server import FastMCP

from src.config import settings
from src.search.engines import get_search_engine
from src.search.models import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


def search_handler(server: FastMCP) -> None:
    """Register the search tool."""

    @server.tool()
    async def search(query: str, max_results: int | None = None) -> str:
        """Search the web. Returns titles, URLs, and snippets.

        Follow up with fetch() to extract content from specific URLs,
        or browser_screenshot(url=...) for a one-off visual.

        Args:
            query: Search query
            max_results: Max results (default from config)

        Returns:
            JSON string of search results
        """
        engine = get_search_engine()
        results = await engine.search(query, max_results)

        # Convert to SearchResult objects with engine and timestamp
        search_results = [
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                engine=settings.SEARCH_ENGINE,
                timestamp=datetime.utcnow(),
            )
            for r in results
        ]

        response = SearchResponse(
            query=query,
            results=search_results,
            total=len(search_results),
        )

        return response.model_dump_json(indent=2)

    logger.info("Registered search tool")
