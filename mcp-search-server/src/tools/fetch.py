"""Fetch tool for MCP server."""

import logging

from mcp.server import Server

from src.browser.automation import browser_manager
from src.extractor.content import ContentExtractor

logger = logging.getLogger(__name__)


def register_fetch_tool(server: Server) -> None:
    """Register the fetch tool with the MCP server.

    Args:
        server: The MCP server instance.
    """

    @server.tool("fetch", "Fetch and extract content from a URL")
    async def fetch_tool(url: str) -> str:
        """Fetch and extract content from a URL.

        Uses browser automation to render JavaScript-heavy pages and
        extracts the main content.

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

            return str(content)
        except Exception as e:
            logger.error("Fetch error: %s", str(e))
            return f"Error fetching URL: {str(e)}"

    logger.info("Registered fetch tool")
