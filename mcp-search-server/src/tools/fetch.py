"""Fetch tool for MCP server."""

import json
import logging

from mcp.server import FastMCP

from src.browser.automation import browser_manager
from src.config import settings
from src.extractor.content import ContentExtractor

logger = logging.getLogger(__name__)


def fetch_handler(server: FastMCP) -> None:
    """Register the fetch tool."""

    @server.tool()
    async def fetch(url: str, truncate: bool = True) -> str:
        """Fetch and extract text content from a URL.

        Renders JavaScript via a headless browser. For multi-step interactions
        (clicking, filling forms), use browser_create_session() + browser tools.

        Args:
            url: URL to fetch
            truncate: When True (default), content exceeding the max length is summarized.
                      When False, the full extracted content is returned without truncation.

        Returns:
            JSON string of extracted content
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Navigate to the URL
            page = await browser_manager.goto(url)

            # Get the rendered page content
            html = await browser_manager.get_content(page)

            # Extract content
            extractor = ContentExtractor()
            content = extractor.extract(html, max_length=settings.FETCH_MAX_LENGTH, truncate=truncate)

            # Clean up page
            await page.close()

            return json.dumps(content, indent=2)
        except Exception as e:
            logger.error("Fetch error: %s", str(e))
            return f"Error fetching URL: {str(e)}"

    logger.info("Registered fetch tool")
