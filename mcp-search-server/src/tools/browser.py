"""Browser automation tools for MCP server.

Consolidated, programmatic interface over Playwright. Instead of many narrow
wrappers (click/fill/get_text/...), the LLM drives the page directly with
async Playwright Python via ``browser_run``. Two helpers remain because they
can't be expressed as a code return value: ``browser_screenshot`` (returns an
MCP image for vision) and ``browser_close`` (session cleanup).
"""

import asyncio
import base64
import contextlib
import io
import json
import logging
import re
import textwrap

from mcp.server import FastMCP
from mcp.types import ImageContent, TextContent

from src.browser.automation import get_browser_manager as _get_browser_manager
from src.config import settings
from src.output_store import output_store

logger = logging.getLogger(__name__)

# Cap on serialized result / captured stdout returned to the model.
_MAX_OUTPUT_CHARS = 50000


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
    async def browser_run(
        code: str,
        session_id: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Drive a real browser by running async Playwright Python.

        This is the primary browser tool: write Playwright code instead of
        calling many small wrappers. The code body runs inside an async function
        with these names already in scope:
          - page    : Playwright Page (await page.goto(url), page.click(sel), ...)
          - context : the BrowserContext (await context.new_page(), cookies, ...)
          - interactables() : async helper -> list of clickable/fillable elements
                              [{index, tag, type, text, name, selector, visible}]
          - mgr     : the BrowserManager (mgr.get_content(page), mgr.screenshot(page))
        Use `return <value>` to send data back; print() output is also captured.

        Sessions: pass a stable session_id (any string, e.g. "main") to keep the
        page/cookies alive across calls. Omit it for a one-off page that is closed
        after the call. Use browser_screenshot for visual checks, browser_close to
        free a session.

        Example:
            await page.goto("https://example.com")
            await page.fill("input[name='q']", "playwright")
            await page.click("button[type='submit']")
            await page.wait_for_load_state("networkidle")
            return await page.title()

        SECURITY: code runs in-process (full Python/host access), gated behind the
        server's API key. Intended for trusted local use.

        Large stdout/string results are previewed with a `stdout_handle` /
        `result_handle`; call read_output(handle=...) to read the rest.
        Returns: {status, result, stdout, url, title, interactables, session_id}
        """
        page = None
        sid = session_id
        one_off = session_id is None
        try:
            page, sid = await _ensure_page(session_id)
            mgr = _get_browser_manager()

            async def interactables():
                return await mgr.get_interactables(page)

            wrapper = (
                "async def __browser_main(page, context, mgr, interactables):\n"
                + textwrap.indent(code or "pass", "    ")
            )
            exec_globals = {"asyncio": asyncio, "json": json, "re": re}
            ns: dict = {}
            exec(compile(wrapper, "<browser_run>", "exec"), exec_globals, ns)

            buf = io.StringIO()
            eff_timeout = timeout or max(settings.BROWSER_TIMEOUT * 2, 60)
            with contextlib.redirect_stdout(buf):
                result = await asyncio.wait_for(
                    ns["__browser_main"](page, page.context, mgr, interactables),
                    timeout=eff_timeout,
                )

            try:
                if isinstance(result, (str, int, float, bool, type(None))):
                    result_repr = result
                else:
                    result_repr = json.loads(json.dumps(result, default=str))
            except Exception:
                result_repr = str(result)

            stdout_full = buf.getvalue()
            try:
                url, title = page.url, await page.title()
            except Exception:
                url, title = None, None
            summary = await _interactables_summary(page)

            response = {
                "status": "success",
                "result": result_repr,
                "url": url,
                "title": title,
                "interactables": summary,
                "session_id": sid,
            }
            # Paginate oversized stdout / string results via read_output.
            output_store.attach(response, "stdout", stdout_full, source="browser_run stdout")
            if isinstance(result_repr, str) and len(result_repr) > settings.OUTPUT_PREVIEW_CHARS:
                output_store.attach(response, "result", result_repr, source="browser_run result")
            return json.dumps(response, indent=2, default=str)
        except Exception as e:
            logger.error("browser_run error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e), "session_id": sid})
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(sid)

    @server.tool()
    async def browser_screenshot(
        url: str | None = None,
        full_page: bool = False,
        session_id: str | None = None,
    ):
        """Capture a screenshot and return it as an MCP image (for vision).

        url: optional page to load first. session_id: screenshot a persistent
        session's current page (created by browser_run). Omit both to shoot the
        one-off page after a url load.
        Returns: MCP ImageContent (base64 PNG) + a file:// resource URI.
        """
        page = None
        one_off = session_id is None
        sid = session_id
        try:
            page, sid = await _ensure_page(session_id)
            if url:
                await page.goto(
                    url,
                    timeout=settings.BROWSER_TIMEOUT * 1000,
                    wait_until="domcontentloaded",
                )

            screenshot_bytes, screenshot_path = await _get_browser_manager().screenshot(
                page, full_page=full_page, session_id=session_id,
            )
            summary = await _interactables_summary(page)

            image = ImageContent(
                type="image",
                data=base64.b64encode(screenshot_bytes).decode("utf-8"),
                mimeType="image/png",
            )
            info = (
                f"Screenshot saved: file://{screenshot_path}\n"
                f"Session: {sid or 'one-off'}  Full page: {full_page}"
            )
            if summary:
                info += "\n" + summary
            return [image, TextContent(type="text", text=info)]
        except Exception as e:
            logger.error("browser_screenshot error: %s", str(e))
            return [TextContent(type="text", text=f"Error: {str(e)}")]
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(sid)

    @server.tool()
    async def browser_close(session_id: str) -> str:
        """Close a browser session and free its pages/cookies.

        Pass the session_id you used with browser_run.
        """
        try:
            ok = await _get_browser_manager().close_session(session_id)
            return json.dumps({
                "status": "success" if ok else "error",
                "message": f"Session {session_id} {'closed' if ok else 'not found'}",
            })
        except Exception as e:
            logger.error("browser_close error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    logger.info("Registered browser automation tools (browser_run, browser_screenshot, browser_close)")
