"""read_output tool: paginate through large outputs from other tools."""

import logging
from typing import Annotated

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field

from src.config import settings
from src.output_store import output_store
from src.tools._report import error_report

logger = logging.getLogger(__name__)


def _fences_as_text(source: str) -> bool:
    """Whether a stored output was originally rendered in a ```text``` fence.

    code_run previews stdout, stderr and (text) result inside fenced code blocks,
    so read_output must fence their continuations too. fetch and deep_search store
    raw markdown, which must stay unfenced.
    """
    return not source.startswith(("fetch", "deep_search"))


def read_output_handler(server: FastMCP) -> None:
    """Register the read_output tool."""

    @server.tool()
    async def read_output(
        handle: Annotated[str, Field(description="The `*_handle` value from a previewing tool's result.")],
        offset: Annotated[
            int,
            Field(description="Start character index; pass the `*_next_offset` from the previous result to continue."),
        ] = 0,
        limit: Annotated[
            int | None,
            Field(description="Characters to return in this window (defaults to ~16000)."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Read more of a large output that another tool only previewed.

        When fetch / deep_search / code_run / take_snapshot / get_content preview
        a big result, they add three sibling fields next to the previewed value,
        e.g. for `content`: `content_handle`, `content_next_offset`, and
        `content_hint` (the hint is a ready-to-run `read_output(...)` call you can
        follow directly). Reading the full content is like reading a file by line
        range.

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
        if ctx:
            await ctx.report_progress(0, 1, f"Reading output {handle}\u2026")
        result = output_store.read(handle, offset=offset, limit=limit or settings.READ_OUTPUT_CHUNK_CHARS)

        if result.get("status") != "success":
            msg = result.get("error", "unknown error")
            hint = (
                "Handle not found — it likely expired (~30 min TTL) or was evicted. "
                "Re-run the original tool (fetch/deep_search/code_run/take_snapshot/"
                "get_content) to get a fresh handle."
            )
            return error_report(msg, hint)

        if ctx:
            await ctx.report_progress(1, 1, "Done")

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

        if _fences_as_text(result.get("source", "")):
            content = f"```text\n{content}\n```"

        return f"{content}\n\n---\n{footer}"

    logger.info("Registered read_output tool")
