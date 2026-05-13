"""Fetch tool for MCP server."""

import json
import logging

from mcp.server import FastMCP

from src.browser.automation import browser_manager
from src.extractor.content import ContentExtractor

logger = logging.getLogger(__name__)


def fetch_handler(server: FastMCP) -> None:
    """Register the fetch tool with the MCP server.

    Args:
        server: The MCP server instance.
    """

    @server.tool()
    async def fetch(url: str) -> str:
        """Fetch and extract content from a URL.

        Use this tool when the user specifies a URL to fetch, or when you need to extract
        content from a specific page. This tool uses browser automation to render
        JavaScript-heavy pages and extracts the main content.

        If this tool fails (e.g., the page requires JavaScript rendering), use browser_navigate()
        and browser_get_content() as a fallback for full browser interaction.

        Args:
            url: The URL to fetch

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
            content = extractor.extract(html)

            # Clean up page
            await page.close()

            return json.dumps(content, indent=2)
        except Exception as e:
            logger.error("Fetch error: %s", str(e))
            return f"Error fetching URL: {str(e)}"

    logger.info("Registered fetch tool")
