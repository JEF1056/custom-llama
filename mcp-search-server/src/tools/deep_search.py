"""Deep search tool for MCP server."""

import logging
from typing import Annotated

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field

from src.browser.automation import browser_manager
from src.config import settings
from src.extractor.content import ContentExtractor
from src.output.format import format_result
from src.output_store import output_store
from src.search.engines import get_search_engine

logger = logging.getLogger(__name__)


def deep_search_handler(server: FastMCP) -> None:
    """Register the deep search tool."""

    @server.tool()
    async def deep_search(
        query: Annotated[str, Field(description="What to search the web for.")],
        max_results: Annotated[
            int | None,
            Field(description="Size of the search pool; full content is extracted from the top 3 only."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Search the web and extract full page content from the top 3 results in one call.

        Slower than search() but returns full text — use when snippets aren't enough.
        max_results: controls search pool; content extracted from top 3 only.
        For interactive browsing, use navigate_page, click, fill, etc.
        Each result is previewed; oversized content exposes a read_output handle in
        its footer so you can read the rest.
        Returns: compact text — numbered results with content, remaining hits as link list.
        """
        engine = get_search_engine()
        if ctx:
            await ctx.report_progress(0, 1, f'Searching for "{query}"\u2026')
        search_results = await engine.search(query, max_results)

        if not search_results:
            return f'No results from {settings.SEARCH_ENGINE}.'

        # Start browser if not running
        if not browser_manager.is_running:
            await browser_manager.start()

        parts: list[str] = [f'Deep search: "{query}"']

        # Extract content from top results
        top = search_results[:3]  # Limit to top 3 for deep extraction
        # Progress total: one step per extracted result plus a final formatting step.
        progress_total = len(top) + 1
        for i, result in enumerate(top, 1):
            title = result.get("title", "") or "(untitled)"
            url = result.get("url", "")
            if ctx:
                await ctx.report_progress(
                    i - 1, progress_total, f"Reading result {i}/{len(top)}: {title}"
                )
            heading = f"[{title}]({url})" if url else title
            parts.append(f"{i}. {heading}")

            # Try to extract content from the URL
            try:
                page = await browser_manager.goto(url)
                html = await browser_manager.get_content(page)

                extractor = ContentExtractor()
                content = extractor.extract(html, truncate="never")
                full_text = content.get("content", "")
                await page.close()

                holder: dict = {}
                output_store.attach(
                    holder,
                    "content",
                    full_text,
                    source=f"deep_search: {url}",
                    inline_chars=2000,
                )
                content_text = holder.get("content", "") or "_(no content extracted)_"
                hint = holder.get("content_hint")
                if hint:
                    parts.append(format_result(content_text, footer=hint))
                else:
                    parts.append(content_text)
            except Exception as e:
                logger.error("Deep search content extraction error for %s: %s", url, str(e))
                parts.append("_(failed to extract content)_")
            parts.append("")

        # Remaining hits as a compact link list
        rest = search_results[3:]
        if rest:
            if ctx:
                await ctx.report_progress(len(top), progress_total, "Formatting remaining hits\u2026")
            parts.append("Other results:")
            for result in rest:
                title = result.get("title", "") or "(untitled)"
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                heading = f"[{title}]({url})" if url else title
                if snippet:
                    parts.append(f"- {heading} — {snippet}")
                else:
                    parts.append(f"- {heading}")

        if ctx:
            await ctx.report_progress(progress_total, progress_total, "Done")
        return "\n".join(parts).rstrip()

    logger.info("Registered deep_search tool")
