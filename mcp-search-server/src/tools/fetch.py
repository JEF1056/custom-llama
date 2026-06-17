"""Fetch tool for MCP server."""

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
        Returns: markdown — title, URL, the extracted content, and (when the page
        was truncated) a footer with the read_output handle to fetch the rest.
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

            title = content.get("title") or "(untitled)"
            preview = content.get("content", "")
            total_chars = len(full_text)

            heading = f"[{title}]({url})" if url else title
            parts = [f"# {heading}", "", f"_Source: {url}_", "", preview]

            # If the inline preview is shorter than the full text, expose a handle
            # so the model can read the remainder via read_output.
            if len(preview) < total_chars:
                handle = output_store.store(full_text, source=f"fetch: {url}")
                parts.append("")
                parts.append("---")
                parts.append(
                    f"_Preview shown ({len(preview)} of {total_chars} chars). "
                    f'Call read_output(handle="{handle}", offset=0) to read the full page._'
                )

            return "\n".join(parts).rstrip()
        except Exception as e:
            logger.error("Fetch error: %s", str(e))
            return f"**Error fetching URL:** {str(e)}"

    logger.info("Registered fetch tool")
