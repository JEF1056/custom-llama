"""Deep search tool for MCP server."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server import Context
from pydantic import Field

from src.browser.automation import browser_manager
from src.config import settings
from src.extractor.content import ContentExtractor
from src.output.format import format_result
from src.output_store import output_store
from src.search.engines import get_search_engine

logger = logging.getLogger(__name__)


@dataclass
class SearchResultItem:
    """Holds extracted content for a single search result."""
    index: int
    title: str
    url: str
    content: str
    error: str | None = None


def _extract_item_content(url: str, index: int) -> SearchResultItem:
    """Synchronously extract content from a single URL.
    
    Called from thread pool to avoid blocking the event loop during
    content extraction (html2text conversion, etc.).
    """
    try:
        # We need to run this in the event loop context since it uses async browser APIs.
        # The caller will use asyncio.to_thread for the outer sync wrapper.
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            page = loop.run_until_complete(browser_manager.goto(url))
            html = loop.run_until_complete(browser_manager.get_content(page))
            loop.run_until_complete(page.close())

            extractor = ContentExtractor()
            full_text = extractor.extract(html, truncate="never").get("content", "")
            return SearchResultItem(index=index, title="", url=url, content=full_text, error=None)
        finally:
            loop.close()
    except Exception as e:
        logger.error("Deep search content extraction error for %s: %s", url, str(e))
        return SearchResultItem(index=index, title="", url=url, content="", error=str(e))


async def _extract_single_result(url: str, index: int) -> SearchResultItem:
    """Extract content from a single URL using the shared browser manager."""
    try:
        page = await browser_manager.goto(url)
        html = await browser_manager.get_content(page)
        await page.close()

        extractor = ContentExtractor()
        full_text = extractor.extract(html, truncate="never").get("content", "")
        return SearchResultItem(index=index, title="", url=url, content=full_text, error=None)
    except Exception as e:
        logger.error("Deep search content extraction error for %s: %s", url, str(e))
        return SearchResultItem(index=index, title="", url=url, content="", error=str(e))


async def _extract_results_parallel(results: list[dict]) -> list[SearchResultItem]:
    """Extract content from multiple URLs in parallel using asyncio.gather."""
    tasks = [_extract_single_result(r["url"], i) for i, r in enumerate(results)]
    return await asyncio.gather(*tasks)


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

        # Extract content from top results in parallel
        top = search_results[:3]  # Limit to top 3 for deep extraction
        
        # Build heading list first (needed for section filtering later)
        headings: list[dict] = []
        for i, result in enumerate(top, 1):
            title = result.get("title", "") or "(untitled)"
            url = result.get("url", "")
            heading = f"[{title}]({url})" if url else title
            parts.append(f"{i}. {heading}")

        # Progress: search done, now extracting (1 step for the parallel batch)
        if ctx:
            await ctx.report_progress(1, 2, f"Reading {len(top)} result(s)\u2026")

        # Extract all top results in parallel
        extracted_items = await _extract_results_parallel(top)

        # Progress total: extraction done + formatting
        progress_total = 2
        for item in extracted_items:
            idx = item.index
            heading = f"[{top[idx - 1].get('title', '')}]({top[idx - 1].get('url', '')})" if top[idx - 1].get("url") else top[idx - 1].get("title", "(untitled)")
            
            if item.error:
                title = top[idx - 1].get("title", "(untitled)")
                parts.append(f"_{title}: (failed to extract content)_")
            else:
                holder: dict = {}
                output_store.attach(
                    holder,
                    "content",
                    item.content,
                    source=f"deep_search: {top[idx - 1].get('url', '')}",
                    inline_chars=2000,
                )
                content_text = holder.get("content", "") or "_(no content extracted)_"
                hint = holder.get("content_hint")
                if hint:
                    parts.append(format_result(content_text, footer=hint))
                else:
                    parts.append(content_text)
            parts.append("")

        # Remaining hits as a compact link list
        rest = search_results[3:]
        if rest:
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
