"""Search tool for MCP server."""

import logging
from datetime import datetime

from mcp.server import Server

from src.config import settings
from src.search.engines import get_search_engine
from src.search.models import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


def register_search_tool(server: Server) -> None:
    """Register the search tool with the MCP server.

    Args:
        server: The MCP server instance.
    """

    @server.tool("search", "Search the web for information")
    async def search_tool(query: str, max_results: int | None = None) -> str:
        """Search the web for information.

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
