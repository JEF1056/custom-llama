"""read_output tool: paginate through large outputs from other tools."""

import json
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
        Keep calling with the returned `next_offset` until `has_more` is false.
        Handles expire after ~30 min; if expired, re-run the original tool.
        Returns: {status, content, offset, returned_chars, total_chars, has_more, next_offset}
        """
        result = output_store.read(handle, offset=offset, limit=limit or settings.READ_OUTPUT_CHUNK_CHARS)
        return json.dumps(result, indent=2, ensure_ascii=False)

    logger.info("Registered read_output tool")
