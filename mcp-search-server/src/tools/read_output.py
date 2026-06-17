"""read_output tool: paginate through large outputs from other tools."""

import logging

from mcp.server import FastMCP

from src.config import settings
from src.output_store import output_store

logger = logging.getLogger(__name__)


def read_output_handler(server: FastMCP) -> None:
    """Register the read_output tool."""

    @server.tool()
    async def read_output(handle: str, offset: int = 0, limit: int | None = None) -> str:
        """Read more of a large output that another tool only previewed.

        When fetch / deep_search / code_run / browser_run preview a big result,
        they add three sibling fields next to the previewed value, e.g. for
        `content`: `content_handle`, `content_next_offset`, and `content_hint`
        (the hint is a ready-to-run `read_output(...)` call you can follow
        directly). Reading the full content is like reading a file by line range.

        handle: the `*_handle` value from the previewing tool's result.
        offset: starting character index — pass the `*_next_offset` from that
                result (or, on subsequent calls, the `next_offset` this tool
                returns) to continue where you left off.
        limit:  characters to return (defaults to config, ~16000).
        Keep calling with the returned `next_offset` until the footer says the end
        is reached. Handles expire after ~30 min; if expired, re-run the original tool.
        Returns: markdown — the content window followed by a footer with the byte
        range read, total size, and the next read_output call (or "end of output").
        """
        result = output_store.read(handle, offset=offset, limit=limit or settings.READ_OUTPUT_CHUNK_CHARS)

        if result.get("status") != "success":
            return f"**Error:** {result.get('error', 'unknown error')}"

        content = result.get("content", "")
        start = result.get("offset", 0)
        returned = result.get("returned_chars", 0)
        total = result.get("total_chars", 0)
        end = start + returned

        if result.get("has_more"):
            next_offset = result.get("next_offset")
            footer = (
                f"_Read chars {start}–{end} of {total}. "
                f'Continue: read_output(handle="{handle}", offset={next_offset})._'
            )
        else:
            footer = f"_Read chars {start}–{end} of {total}. End of output._"

        return f"{content}\n\n---\n{footer}"

    logger.info("Registered read_output tool")
