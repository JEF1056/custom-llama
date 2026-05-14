"""Deep search tool for MCP server."""

import json
import logging
from datetime import datetime

from mcp.server import FastMCP

from src.browser.automation import browser_manager
from src.config import settings
from src.extractor.content import ContentExtractor
from src.search.engines import get_search_engine
from src.search.models import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


def deep_search_handler(server: FastMCP) -> None:
    """Register the deep search tool."""

    @server.tool()
    async def deep_search(query: str, max_results: int | None = None) -> str:
        """Search the web and extract content from top 3 results.

        Use for comprehensive information in a single call. For interactive
        browsing, use browser_create_session() + browser tools.

        Args:
            query: Search query
            max_results: Max results (default from config)

        Returns:
            JSON string with search results and extracted content from top 3
        """
        engine = get_search_engine()
        search_results = await engine.search(query, max_results)

        # Start browser if not running
        if not browser_manager.is_running:
            await browser_manager.start()

        # Extract content from top results
        enriched_results: list[dict] = []
        for i, result in enumerate(search_results[:3]):  # Limit to top 3 for deep extraction
            enriched = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
            }

            # Try to extract content from the URL
            try:
                page = await browser_manager.goto(result.get("url", ""))
                html = await browser_manager.get_content(page)

                extractor = ContentExtractor()
                content = extractor.extract(html)
                enriched["content"] = content.get("content", "")[:2000]  # Limit content length
                enriched["content_length"] = len(content.get("content", ""))

                await page.close()
            except Exception as e:
                logger.error("Deep search content extraction error for %s: %s", result.get("url", ""), str(e))
                enriched["content"] = ""
                enriched["content_length"] = 0

            enriched_results.append(enriched)

        # Convert search results to SearchResult objects
        search_result_objects = [
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                engine=settings.SEARCH_ENGINE,
                timestamp=datetime.utcnow(),
            )
            for r in search_results
        ]

        response = SearchResponse(
            query=query,
            results=search_result_objects,
            total=len(search_results),
        )

        # Add deep search results
        response_dict = response.model_dump()
        response_dict["deep_results"] = enriched_results

        return json.dumps(response_dict, indent=2)

    logger.info("Registered deep_search tool")
