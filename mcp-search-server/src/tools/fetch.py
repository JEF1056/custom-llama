"""Fetch tool for MCP server."""

import json
import logging

from mcp.server import FastMCP

from src.browser.automation import browser_manager
from src.config import settings
from src.extractor.content import ContentExtractor, TruncationMode
from src.output_store import output_store

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

        Long pages return a condensed preview plus a `content_handle`; call
        read_output(handle=content_handle) to read the full text in windows.
        Returns: {url, title, content, content_length, truncated,
        content_handle?, content_total_chars?, content_next_offset?, content_hint?}
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Navigate to the URL
            page = await browser_manager.goto(url)

            # Get the rendered page content
            html = await browser_manager.get_content(page)

            # Extract the inline preview (honours the caller's truncate mode)
            extractor = ContentExtractor()
            content = extractor.extract(
                html,
                max_length=settings.FETCH_MAX_LENGTH,
                truncate=truncate,
                code_block_max_chars=code_block_max_chars,
            )

            # Extract the full, untruncated text for paginated reads
            full_extractor = ContentExtractor()
            full_content = full_extractor.extract(html, truncate="never")
            full_text = full_content["content"]

            # Filter to specific sections if requested
            if sections:
                content["content"] = extractor._extract_sections(
                    content["content"], content["headings"], sections
                )
                full_text = full_extractor._extract_sections(
                    full_text, full_content["headings"], sections
                )

            # Clean up page
            await page.close()

            content["content_length"] = len(full_text)
            # If the inline preview is shorter than the full text, expose a handle
            # so the model can read the remainder via read_output.
            if len(content["content"]) < len(full_text):
                handle = output_store.store(full_text, source=f"fetch: {url}")
                content["truncated"] = True
                content["content_handle"] = handle
                content["content_total_chars"] = len(full_text)
                # Preview is a summary (not a prefix), so the full text reads from 0.
                content["content_next_offset"] = 0
                content["content_hint"] = (
                    f"Preview shown ({len(content['content'])} of {len(full_text)} chars). "
                    f'Call read_output(handle="{handle}", offset=0) to read the full page.'
                )
            else:
                content["truncated"] = False

            return json.dumps(content, indent=2)
        except Exception as e:
            logger.error("Fetch error: %s", str(e))
            return f"Error fetching URL: {str(e)}"

    logger.info("Registered fetch tool")
