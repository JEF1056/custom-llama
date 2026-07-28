"""Browser automation tools for MCP server.

Discrete, fine-grained Playwright wrappers for common browser actions:
navigate_page, click, fill, get_text, evaluate, get_interactables, get_content,
take_snapshot, page_state, and browser_screenshot for visual context.
"""

import asyncio
import base64
import contextlib
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Annotated

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import ImageContent, TextContent
from pydantic import Field

from src.browser.automation import get_browser_manager as _get_browser_manager
from src.config import settings
from src.output.format import format_result
from src.output_store import output_store

logger = logging.getLogger(__name__)


# Unit Separator — won't appear in a model-chosen session_id, so it safely
# delimits the per-caller namespace from the caller's own id.
_NS_DELIM = "\x1f"


def _client_ns(ctx: Context | None) -> str | None:
    """Return a stable token identifying the calling connection, or None.

    Two unrelated agents talking to this server concurrently may both pick the
    same session_id (e.g. "wm1"); without isolation they would share one browser
    context and clobber each other. We key each caller's sessions under a token
    derived from their transport connection. Preference order:
      1. the stateful StreamableHTTP ``mcp-session-id`` (a per-connection UUID,
         never reused) read from the request headers,
      2. the client-declared ``client_id`` from the request meta,
      3. the identity of the live ServerSession object (stable while connected).
    Returns None when no caller identity is available (e.g. stdio / single
    tenant), in which case ids are used as-is (legacy behaviour).
    """
    if ctx is None:
        return None
    try:
        rc = ctx.request_context
    except Exception:
        return None
    req = getattr(rc, "request", None)
    if req is not None:
        with contextlib.suppress(Exception):
            header_sid = req.headers.get("mcp-session-id")
            if header_sid:
                return header_sid
    with contextlib.suppress(Exception):
        if ctx.client_id:
            return str(ctx.client_id)
    with contextlib.suppress(Exception):
        return f"obj{id(rc.session)}"
    return None


def _scope_id(ctx: Context | None, session_id: str | None) -> str | None:
    """Map a caller-facing session_id to a globally-unique internal key."""
    if session_id is None:
        return None
    ns = _client_ns(ctx)
    return f"{ns}{_NS_DELIM}{session_id}" if ns else session_id


def _unscope_id(ctx: Context | None, internal_id: str) -> str | None:
    """Inverse of :func:`_scope_id` for THIS caller.

    Returns the caller-facing id if ``internal_id`` belongs to this connection,
    else None (so a caller never sees another connection's sessions).
    """
    ns = _client_ns(ctx)
    if not ns:
        # No namespace available: only legacy, un-delimited ids are "ours".
        return None if _NS_DELIM in internal_id else internal_id
    prefix = f"{ns}{_NS_DELIM}"
    return internal_id[len(prefix):] if internal_id.startswith(prefix) else None


async def _ensure_page(session_id: str | None):
    """Return a live page.

    session_id=None  -> ephemeral one-off page in a transient session (caller closes it).
    session_id="x"   -> persistent named session, created on first use and reused after.
    """
    mgr = _get_browser_manager()
    if not mgr.is_running:
        await mgr.start()

    if session_id is None:
        # Transient session for a single call; closed by the caller.
        sid = await mgr.create_session()
        session = mgr._sessions[sid]
        page = await session.context.new_page()
        session.pages.append(page)
        return page, sid

    session = mgr._sessions.get(session_id)
    if session is None:
        await mgr.create_session(session_id=session_id)
        session = mgr._sessions[session_id]
    else:
        mgr._touch_session(session_id)

    if session.pages:
        return session.pages[-1], session_id
    page = await session.context.new_page()
    session.pages.append(page)
    return page, session_id


async def _interactables_summary(page) -> str:
    """Compact list of visible clickable/fillable elements with selectors."""
    try:
        elements = await _get_browser_manager().get_interactables(page)
    except Exception:
        return ""
    visible = [e for e in elements if e.get("visible")]
    if not visible:
        return ""
    lines = [f"Interactables ({len(visible)} visible / {len(elements)} total):"]
    for el in visible[:40]:
        tag = el.get("tag", "")
        etype = el.get("type", "")
        label = next((p for p in (el.get("text", ""), el.get("name", ""),
                                  el.get("placeholder", "")) if p), f"<{tag}>")[:80]
        lines.append(f"  [{el['index']}] {etype} {tag}: \"{label}\" -> {el.get('selector', '')}")
    if len(visible) > 40:
        lines.append(f"  ... and {len(visible) - 40} more")
    return "\n".join(lines)


def browser_handler(server: FastMCP) -> None:
    """Register the consolidated browser automation tools."""

    @server.tool()
    async def browser_screenshot(
        url: Annotated[
            str | None,
            Field(description="Optional page to load before capturing."),
        ] = None,
        full_page: Annotated[
            bool,
            Field(description="Capture the entire scrollable page instead of just the viewport."),
        ] = False,
        session_id: Annotated[
            str | None,
            Field(description="Screenshot the current page of a persistent session."),
        ] = None,
        ctx: Context | None = None,
    ):
        """Capture a screenshot and return it as an MCP image (for vision).

        url: optional page to load first. session_id: screenshot a persistent
        session's current page. Omit both to shoot the one-off page after a
        url load.
        Returns: MCP ImageContent (base64 PNG) + a file:// resource URI.
        """
        page = None
        one_off = session_id is None
        sid = session_id
        try:
            if ctx:
                await ctx.report_progress(0, 2, "Preparing page\u2026")
            page, sid = await _ensure_page(session_id)
            if url:
                await page.goto(
                    url,
                    timeout=settings.BROWSER_TIMEOUT * 1000,
                    wait_until="domcontentloaded",
                )

            if ctx:
                await ctx.report_progress(1, 2, "Capturing screenshot\u2026")
            screenshot_bytes, screenshot_path = await _get_browser_manager().screenshot(
                page, full_page=full_page, session_id=session_id,
            )
            summary = await _interactables_summary(page)

            image = ImageContent(
                type="image",
                data=base64.b64encode(screenshot_bytes).decode("utf-8"),
                mimeType="image/png",
            )
            # Build a publicly fetchable URL so the chat UI can render the image.
            # screenshot_path lives under FILE_OUTPUT_DIR (e.g.
            # /app/mcp-files/screenshots/foo.png); expose it via the /files route.
            try:
                rel_path = Path(screenshot_path).resolve().relative_to(
                    Path(settings.FILE_OUTPUT_DIR).resolve()
                )
                public_url = f"{settings.FILE_BASE_URL.rstrip('/')}/files/{rel_path.as_posix()}"
            except ValueError:
                public_url = None
            info = (
                f"Screenshot URL: {public_url}\n" if public_url
                else f"Screenshot saved: file://{screenshot_path}\n"
            )
            info += f"Session: {sid or 'one-off'}  Full page: {full_page}"
            if summary:
                info += "\n" + summary
            if ctx:
                await ctx.report_progress(2, 2, "Done")
            return [image, TextContent(type="text", text=info)]
        except Exception as e:
            logger.error("browser_screenshot error: %s", str(e))
            return [TextContent(type="text", text=f"Error: {str(e)}")]
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(sid)

    # ── Fine-grained browser tools ──────────────────────────────────────────

    @server.tool()
    async def navigate_page(
        url: Annotated[str, Field(description="The URL to navigate to.")],
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off navigation."),
        ] = None,
        wait_until: Annotated[
            str | None,
            Field(description="When to consider navigation complete. Default: domcontentloaded"),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Navigate to a URL and return page state.

        Returns: url, title, accessibility snapshot, and interactables count.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            if ctx:
                await ctx.report_progress(0, 2, f"Navigating to {url}\u2026")
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            wait = wait_until or "domcontentloaded"
            await page.goto(url, timeout=settings.BROWSER_TIMEOUT * 1000, wait_until=wait)
            if ctx:
                await ctx.report_progress(1, 2, "Extracting page state\u2026")
            state = await mgr.get_page_state(page, max_length=8000)
            snapshot = state.get("snapshot", "")
            interactables_count = state.get("interactables_count", 0)
            lines = [f"URL: {state.get('url', '')}", f"Title: {state.get('title', '')}"]
            if snapshot:
                lines.append(format_result(snapshot, footer=f"Interactables: {interactables_count} visible"))
            else:
                lines.append(f"Interactables: {interactables_count} visible")
            return "\n".join(lines)
        except Exception as e:
            logger.error("navigate_page error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def take_snapshot(
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        depth: Annotated[
            int | None,
            Field(description="Maximum ARIA tree depth. None for full. Recommended: 3 for page recon."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Take an ARIA accessibility snapshot of the current page.

        Returns a structured text representation of the page with [ref=eN] markers
        for element targeting. Use mode='ai' for LLM consumption.

        Large snapshots are paginated via read_output.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            if ctx:
                await ctx.report_progress(0, 2, "Taking snapshot\u2026")
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            snapshot = await mgr.get_accessibility_snapshot(page, depth=depth)
            if ctx:
                await ctx.report_progress(1, 2, "Done")

            # Paginate large snapshots
            holder: dict = {}
            output_store.attach(holder, "snapshot", snapshot, source="take_snapshot")
            result_text = holder.get("snapshot", "")
            hint = holder.get("snapshot_hint")

            if hint:
                return format_result(result_text, footer=hint)
            return result_text
        except Exception as e:
            logger.error("take_snapshot error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def page_state(
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Get unified page state: URL, title, accessibility snapshot, interactables, scroll.

        Returns compact text with URL, title, snapshot, and interactables info.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            if ctx:
                await ctx.report_progress(0, 2, "Getting page state\u2026")
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            state = await mgr.get_page_state(page, max_length=8000)
            if ctx:
                await ctx.report_progress(1, 2, "Done")
            lines = [f"URL: {state.get('url', '')}", f"Title: {state.get('title', '')}"]
            snapshot = state.get("snapshot", "")
            interactables_count = state.get("interactables_count", 0)
            if snapshot:
                lines.append(format_result(snapshot, footer=f"Interactables: {interactables_count} visible"))
            else:
                lines.append(f"Interactables: {interactables_count} visible")
            return "\n".join(lines)
        except Exception as e:
            logger.error("page_state error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def click(
        selector: Annotated[
            str,
            Field(
                description="CSS selector for the element to click. "
                "Use interactables() to discover selectors first."
            ),
        ],
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Click an element on the current page by CSS selector.

        Returns compact text: selector, url, title, interactables count.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            if ctx:
                await ctx.report_progress(0, 3, f"Preparing page\u2026")
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            if ctx:
                await ctx.report_progress(1, 3, f"Clicking {selector}\u2026")
            await mgr.click(page, selector)
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
            interactables = await mgr.get_interactables(page)
            if ctx:
                await ctx.report_progress(2, 3, "Done")
            return format_result(
                f"Clicked: {selector}\nURL: {page.url} | Title: {await page.title()}",
                footer=f"Interactables: {len(interactables)} visible",
            )
        except Exception as e:
            logger.error("click error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def fill(
        selector: Annotated[
            str,
            Field(description="CSS selector for the input element to fill."),
        ],
        value: Annotated[
            str,
            Field(description="The value to fill into the input."),
        ],
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Fill an input field on the current page by CSS selector.

        Returns compact text: selector, value, url, title.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            if ctx:
                await ctx.report_progress(0, 3, "Preparing page\u2026")
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            if ctx:
                await ctx.report_progress(1, 3, f"Filling {selector}\u2026")
            await mgr.fill(page, selector, value)
            if ctx:
                await ctx.report_progress(2, 3, "Done")
            return f"Filled: {selector} = \"{value}\"\nURL: {page.url} | Title: {await page.title()}"
        except Exception as e:
            logger.error("fill error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def get_text(
        selector: Annotated[
            str,
            Field(description="CSS selector for the element whose text to read."),
        ],
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Get the inner text of an element by CSS selector.

        Returns the raw text content directly.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            text = await mgr.get_text(page, selector)
            return text
        except Exception as e:
            logger.error("get_text error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def evaluate(
        script: Annotated[
            str,
            Field(description="JavaScript code to evaluate on the page."),
        ],
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Evaluate JavaScript on the current page.

        Returns the result value, or compact JSON if the result is structured data.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            result = await mgr.evaluate(page, script)
            # If result is a dict or list, return compact JSON to preserve structure
            if isinstance(result, (dict, list)):
                return json.dumps(result, separators=(",", ":"), default=str)
            return str(result)
        except Exception as e:
            logger.error("evaluate error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def get_interactables(
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Get all clickable and fillable elements on the current page.

        Returns a compact list: "[index] type tag: label -> selector" per line.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            elements = await mgr.get_interactables(page)
            visible = [e for e in elements if e.get("visible")]
            if not visible:
                return "No interactable elements found."
            lines = [f"{e['index']} {e.get('type', '')} {e.get('tag', '')}: {next((p for p in (e.get('text', ''), e.get('name', ''), e.get('placeholder', '')) if p), '')[:60]} -> {e.get('selector', '')}" for e in visible[:50]]
            if len(visible) > 50:
                lines.append(f"... and {len(visible) - 50} more")
            return "\n".join(lines)
        except Exception as e:
            logger.error("get_interactables error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    @server.tool()
    async def get_content(
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Omit for one-off."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Get the rendered page content as markdown text.

        Returns the page content converted to markdown via html2text.
        Large outputs are paginated via read_output.
        """
        page = None
        scoped_sid = _scope_id(ctx, session_id) if session_id else None
        one_off = session_id is None
        try:
            page, scoped_sid = await _ensure_page(scoped_sid)
            mgr = _get_browser_manager()
            content = await mgr.get_content(page)
            holder: dict = {}
            output_store.attach(holder, "content", content, source="browser_get_content")
            result_text = holder.get("content", "")
            hint = holder.get("content_hint")
            if hint:
                return format_result(result_text, footer=hint)
            return result_text
        except Exception as e:
            logger.error("get_content error: %s", str(e))
            return f"Error: {str(e)}"
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(scoped_sid)

    logger.info(
        "Registered browser automation tools "
        "(browser_screenshot, navigate_page, take_snapshot, page_state, "
        "click, fill, get_text, evaluate, get_interactables, "
        "get_content)"
    )
