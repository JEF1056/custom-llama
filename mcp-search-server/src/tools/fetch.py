"""Fetch tool for MCP server."""

import json
import logging

from mcp.server import FastMCP

from src.browser.automation import browser_manager
from src.config import settings
from src.extractor.content import ContentExtractor, TruncationMode

logger = logging.getLogger(__name__)


def fetch_handler(server: FastMCP) -> None:
    """Register the fetch tool."""

    @server.tool()
    async def fetch(
        url: str,
        truncate: TruncationMode = "always",
        code_block_max_chars: int | None = None,
        sections: list[str] | None = None,
    ) -> str:
        """Fetch and extract text from a URL (renders JS via headless browser).

        For multi-step interactions (click, fill, scroll) use browser tools instead.
        truncate: "always" (default) | "never" | "main_only" | "code_only"
        sections: list of heading texts → extract only those sections (useful for long pages).
        code_block_max_chars: override per-code-block char limit.
        Returns: {url, title, content, code_blocks, truncated}
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
            content = extractor.extract(
                html,
                max_length=settings.FETCH_MAX_LENGTH,
                truncate=truncate,
                code_block_max_chars=code_block_max_chars,
            )

            # Filter to specific sections if requested
            if sections:
                content["content"] = extractor._extract_sections(
                    content["content"], content["headings"], sections
                )

            # Clean up page
            await page.close()

            return json.dumps(content, indent=2)
        except Exception as e:
            logger.error("Fetch error: %s", str(e))
            return f"Error fetching URL: {str(e)}"

    logger.info("Registered fetch tool")
