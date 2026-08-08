"""Fetch tool for MCP server."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server import Context
from pydantic import Field

from src.browser.automation import browser_manager
from src.config import settings
from src.extractor.content import ContentExtractor, TruncationMode
from src.output.format import format_result
from src.output_store import output_store

logger = logging.getLogger(__name__)


def fetch_handler(server: FastMCP) -> None:
    """Register the fetch tool."""

    @server.tool()
    async def fetch(
        url: Annotated[str, Field(description="The page URL to fetch and extract text from.")],
        truncate: Annotated[
            TruncationMode,
            Field(description='How to trim long pages: "always" (default) | "never" | "main_only" | "code_only".'),
        ] = "always",
        code_block_max_chars: Annotated[
            int | None,
            Field(description="Override the per-code-block character limit."),
        ] = None,
        sections: Annotated[
            list[str] | None,
            Field(description="Heading texts to extract only those sections (useful for long pages)."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Fetch and extract text from a URL (renders JS via headless browser).

        For multi-step interactions (click, fill, scroll) use browser tools instead.
        truncate: "always" (default) | "never" | "main_only" | "code_only"
        sections: list of heading texts → extract only those sections (useful for long pages).
        code_block_max_chars: override per-code-block char limit.

        Long pages return a condensed preview plus a read_output handle in the footer.
        Returns: compact text — title with URL, extracted content, and pagination hint.
        """
        sid: str | None = None
        try:
            # Create an ephemeral session so we can reliably close both page and
            # context when done — just closing the page leaves the context alive.
            # create_session() starts the browser pool if it isn't running yet.
            if ctx:
                await ctx.report_progress(0, 4, "Starting browser\u2026")
            sid = await browser_manager.create_session()

            # Navigate to the URL
            if ctx:
                await ctx.report_progress(1, 4, f"Loading {url}\u2026")
            page = await browser_manager.goto(url, session_id=sid)

            # Get the rendered page content
            html = await browser_manager.get_content(page)

            # Extract the inline preview (honours the caller's truncate mode)
            if ctx:
                await ctx.report_progress(2, 4, "Extracting content\u2026")
            extractor = ContentExtractor()
            content = extractor.extract(
                html,
                max_length=settings.FETCH_MAX_LENGTH,
                truncate=truncate,
                code_block_max_chars=code_block_max_chars,
            )

            # Extract the full, untruncated text for paginated reads
            full_text = extractor.extract(html, truncate="never")["content"]

            # Filter to specific sections if requested
            if sections:
                content["content"] = extractor._extract_sections(
                    content["content"], content["headings"], sections
                )
                full_text = extractor._extract_sections(
                    full_text, content["headings"], sections
                )

            if ctx:
                await ctx.report_progress(3, 4, "Formatting result\u2026")
            title = content.get("title") or "(untitled)"
            preview = content.get("content", "")
            total_chars = len(full_text)

            # If the inline preview is shorter than the full text, expose a handle
            # so the model can read the remainder via read_output.
            if len(preview) < total_chars:
                handle = output_store.store(full_text, source=f"fetch: {url}")
                return format_result(
                    f"[{title}]({url})\n\n{preview}",
                    footer=f"Preview: {len(preview)}/{total_chars} — read_output(handle=\"{handle}\", offset=0)",
                )

            if ctx:
                await ctx.report_progress(4, 4, "Done")
            return f"[{title}]({url})\n\n{preview}"
        except Exception as e:
            logger.error("Fetch error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if sid is not None:
                await browser_manager.close_session(sid)

    logger.info("Registered fetch tool")
