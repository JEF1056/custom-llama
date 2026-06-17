"""Search tool for MCP server."""

import logging
from typing import Annotated

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field

from src.config import settings
from src.search.engines import get_search_engine

logger = logging.getLogger(__name__)


def search_handler(server: FastMCP) -> None:
    """Register the search tool."""

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="What to search the web for.")],
        max_results: Annotated[
            int | None,
            Field(description="Max results to return; defaults to the server config (typically 10)."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Search the web. Returns titles, URLs, and snippets (no full page content).

        max_results: defaults to config value (typically 10).
        Follow up with fetch(url) to get full content, or browser_screenshot(url) for a visual.
        Returns: markdown — a numbered list of results (title, URL, snippet).
        """
        if ctx:
            await ctx.report_progress(0, 1, f'Searching for "{query}"\u2026')
        engine = get_search_engine()
        results = await engine.search(query, max_results)

        if not results:
            if ctx:
                await ctx.report_progress(1, 1, "No results")
            return f'# Search: "{query}"\n\n_No results from {settings.SEARCH_ENGINE}._'

        if ctx:
            await ctx.report_progress(1, 1, f"Found {len(results)} result(s)")
        lines = [
            f'# Search: "{query}"',
            "",
            f"_{len(results)} result(s) from {settings.SEARCH_ENGINE}_",
            "",
        ]
        for i, r in enumerate(results, 1):
            title = r.get("title", "") or "(untitled)"
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            heading = f"[{title}]({url})" if url else title
            lines.append(f"{i}. **{heading}**")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines).rstrip()

    logger.info("Registered search tool")
