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
    """Register the search tool with the MCP server.

    Args:
        server: The MCP server instance.
    """

    @server.tool()
    async def search(query: str, max_results: int | None = None) -> str:
        """Search the web for information using a search engine.

        Use this tool when you need to find information on a topic or when the agent is stuck
        and needs additional context. This is the first step in the research workflow.

        After getting search results, use fetch() to extract content from specific URLs,
        or browser_navigate() for JavaScript-heavy pages that require full browser rendering.

        Args:
            query: The search query
            max_results: Maximum number of results (default from config)

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
