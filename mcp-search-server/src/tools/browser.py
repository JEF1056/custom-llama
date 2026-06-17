"""Browser automation tools for MCP server.

Consolidated, programmatic interface over Playwright. Instead of many narrow
wrappers (click/fill/get_text/...), the LLM drives the page directly with
async Playwright Python via ``browser_run``. Two helpers remain because they
can't be expressed as a code return value: ``browser_screenshot`` (returns an
MCP image for vision) and ``browser_close`` (session cleanup).
"""

import ast
import asyncio
import base64
import contextlib
import io
import json
import logging
import re
import textwrap
import time
import traceback
from typing import Annotated

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import ImageContent, TextContent
from pydantic import Field

from src.browser.automation import get_browser_manager as _get_browser_manager
from src.config import settings
from src.output_store import output_store

logger = logging.getLogger(__name__)


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


def _clean_browser_traceback(exc: BaseException) -> str:
    """Render a traceback whose line numbers match the user's `code`.

    The user's code is wrapped in a 1-line `async def __browser_main(...)` header
    and indented before being compiled under the filename ``<browser_run>``. We
    keep only the frames from that synthetic file, subtract the wrapper's 1-line
    offset so reported line numbers line up with the code the model wrote, and
    fall back to the raw exception text if no user frame is present.
    """
    try:
        tb = exc.__traceback__
        frames = [f for f in traceback.extract_tb(tb) if f.filename == "<browser_run>"]
        lines = [f"{type(exc).__name__}: {exc}".strip()]
        for f in frames:
            # wrapper adds 1 header line before the user's (1-indexed) code.
            user_lineno = (f.lineno or 1) - 1
            lines.append(f"  at code line {user_lineno}")
        return "\n".join(lines)
    except Exception:
        return f"{type(exc).__name__}: {exc}".strip()


def _failure_hint(exc: BaseException) -> str | None:
    """Return a one-line corrective hint for common browser_run failures."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name == "TimeoutError" or "timeout" in msg:
        return (
            "Timed out. Try wait_until=\"domcontentloaded\" instead of \"networkidle\", "
            "raise the timeout=<seconds> arg, or pick a selector from the Interactables "
            "list. If unsure what the page is doing, browser_screenshot (same session_id) "
            "to see it."
        )
    if "strict mode" in msg or "no node found" in msg or "waiting for selector" in msg \
            or "failed to find element" in msg or "resolved to 0 elements" in msg:
        return (
            "Selector did not match. Call interactables() first and use one of the "
            "returned selectors, or make the selector more specific."
        )
    if "has been closed" in msg or "target page" in msg or "target closed" in msg:
        return (
            "The page/context was closed (one-off pages close after the call). "
            "Re-run with a stable session_id to keep state alive across calls."
        )
    return None


# Map common Playwright/page method names to human-readable progress verbs.
# Used to turn each top-level statement of the user's `code` into a live label.
# Verbs ending in a preposition expect a target string (a selector/url); the
# preposition is dropped when no string argument is present.
_STEP_VERBS = {
    "goto": "Navigating to",
    "click": "Clicking",
    "dblclick": "Double-clicking",
    "fill": "Filling",
    "type": "Typing into",
    "press": "Pressing key in",
    "select_option": "Selecting option in",
    "check": "Checking",
    "uncheck": "Unchecking",
    "hover": "Hovering over",
    "focus": "Focusing",
    "tap": "Tapping",
    "set_input_files": "Uploading file to",
    "scroll_into_view_if_needed": "Scrolling to",
    "wait_for_selector": "Waiting for",
    "wait_for_url": "Waiting for navigation",
    "wait_for_load_state": "Waiting for page load",
    "wait_for_timeout": "Waiting",
    "wait_for_event": "Waiting for event",
    "get_content": "Extracting page content",
    "content": "Extracting page content",
    "inner_text": "Reading text from",
    "text_content": "Reading text from",
    "get_attribute": "Reading attribute of",
    "title": "Reading page title",
    "url": "Reading page URL",
    "screenshot": "Capturing screenshot",
    "evaluate": "Running JavaScript",
    "evaluate_handle": "Running JavaScript",
    "reload": "Reloading page",
    "go_back": "Going back",
    "go_forward": "Going forward",
    "new_page": "Opening new page",
    "set_viewport_size": "Resizing viewport",
    "add_cookies": "Setting cookies",
    "get_interactables": "Discovering interactables",
    "interactables": "Discovering interactables",
}


def _stmt_primary_call(node: ast.stmt) -> ast.Call | None:
    """Return the Call that best characterizes a statement, if any."""
    val = None
    if isinstance(node, (ast.Expr, ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign)):
        val = node.value
    if isinstance(val, ast.Await):
        val = val.value
    return val if isinstance(val, ast.Call) else None


def _call_method_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _first_str_arg(call: ast.Call) -> str | None:
    """First string literal argument, searched through chained calls."""
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
        return _first_str_arg(f.value)
    return None


def _short(text: str, limit: int = 48) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _describe_stmt(node: ast.stmt) -> str:
    """Human-readable progress label for one top-level statement of `code`."""
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return "Looping\u2026"
    if isinstance(node, ast.If):
        return "Branching\u2026"
    if isinstance(node, (ast.With, ast.AsyncWith, ast.Try)):
        return "Running block\u2026"
    call = _stmt_primary_call(node)
    if call is not None:
        name = _call_method_name(call)
        verb = _STEP_VERBS.get(name)
        if verb is not None:
            arg = _first_str_arg(call)
            if arg:
                return _short(f"{verb} {arg}")
            for prep in (" to", " into", " in", " of", " for", " over"):
                if verb.endswith(prep):
                    return verb[: -len(prep)]
            return verb
        if isinstance(node, ast.Return):
            return "Returning result"
        return f"Calling {name}()" if name else "Running step"
    if isinstance(node, ast.Return):
        return "Returning result"
    return "Running step"


def _instrument_browser_code(code: str):
    """Compile the user's `code` into async `__browser_main`, injecting a
    progress callback before each top-level statement.

    Returns ``(code_object, step_labels)``. The wrapper adds exactly one header
    line before the user's body so traceback line numbers stay aligned (see
    ``_clean_browser_traceback``). Each injected ``await __report(i)`` is
    copy_location'd onto the statement it precedes, so it shares that line and
    does not shift the mapping. ``__report`` resolves the label from
    ``step_labels`` at runtime.
    """
    wrapper_src = (
        "async def __browser_main(page, context, mgr, interactables, __report):\n"
        + textwrap.indent(code or "pass", "    ")
    )
    tree = ast.parse(wrapper_src)
    func = tree.body[0]
    labels: list[str] = []
    new_body: list[ast.stmt] = []
    for node in func.body:
        idx = len(labels)
        labels.append(_describe_stmt(node))
        report = ast.Expr(value=ast.Await(value=ast.Call(
            func=ast.Name(id="__report", ctx=ast.Load()),
            args=[ast.Constant(idx)],
            keywords=[],
        )))
        ast.copy_location(report, node)
        ast.fix_missing_locations(report)
        new_body.append(report)
        new_body.append(node)
    func.body = new_body or [ast.Pass()]
    ast.fix_missing_locations(tree)
    return compile(tree, "<browser_run>", "exec"), labels


def _render_run_report(
    *,
    status: str,
    sid: str | None,
    title: str | None,
    url: str | None,
    body_parts: list[str],
    stdout_preview: str,
    stdout_hint: str | None,
    summary: str,
    error_block: str | None = None,
    hint: str | None = None,
) -> str:
    """Assemble the markdown report shared by the success and error paths.

    Both paths show the same shape (status line, page line, stdout, interactables)
    so the model always has page state and selectors to recover from — even on
    failure. `body_parts` carries path-specific content (the result on success);
    `error_block`/`hint` are only set on the error path.
    """
    parts = [f"**Status:** {status} · Session: {sid}"]
    if title or url:
        page_line = f"**Page:** {title or '(untitled)'}"
        if url:
            page_line += f" — [{url}]({url})"
        parts.append(page_line)

    if error_block:
        parts += ["", "**Error:**", "```text", error_block, "```"]
        if hint:
            parts.append(f"_{hint}_")

    parts += body_parts

    if stdout_preview:
        parts += ["", "**stdout:**", "```text", stdout_preview, "```"]
        if stdout_hint:
            parts.append(f"_{stdout_hint}_")

    if summary:
        parts += ["", "**Interactables:**", "```text", summary, "```"]

    return "\n".join(parts)


def browser_handler(server: FastMCP) -> None:
    """Register the consolidated browser automation tools."""

    @server.tool()
    async def browser_run(
        code: Annotated[str, Field(description="Async Playwright Python. `page`, `context`, `browser` are in scope; assign `result` or print to return data.")],
        session_id: Annotated[
            str | None,
            Field(description="Persistent session id. Keep it SHORT to save tokens but UNIQUE per concurrent task — a 2-4 char topic hint + a digit, e.g. 'wm1', 'amz2' (a bare word collides; a long random hash wastes tokens). Run independent jobs in parallel by emitting several browser_run calls in ONE turn, each with its own id; omit for a one-off page."),
        ] = None,
        timeout: Annotated[
            int | None,
            Field(description="Max seconds for the code to run (default from config). Raise for slow pages."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Drive a real browser by running async Playwright Python.

        This is the primary browser tool: write Playwright code instead of
        calling many small wrappers. The code body runs inside an async function
        with these names already in scope:
          - page    : Playwright Page (await page.goto(url), page.click(sel), ...)
          - context : the BrowserContext (await context.new_page(), cookies, ...)
          - interactables() : async helper -> list of clickable/fillable elements
                              [{index, tag, type, text, name, selector, visible}].
                              Call this to DISCOVER selectors before clicking/filling.
          - mgr     : the BrowserManager. mgr.get_content(page) -> cleaned page text
                      (the standard way to extract a rendered page).
        Also pre-imported: asyncio, json, re.

        How to write the code:
          - The body is async — `await` every Playwright call.
          - `return <value>` to send data back (str/number/dict/list are fine).
          - `print(...)` is also captured and shown as stdout.
          - To read a JS-rendered page: goto, wait, then `return await mgr.get_content(page)`.
          - To interact: discover with `interactables()`, then click/fill its selectors.

        First call on an unfamiliar page — DON'T blind-guess selectors. Every
        response (success OR error) includes the page's live Interactables list and
        title/URL, so make the first call a cheap recon step, then act with known
        selectors on the next call:
            # call 1 (recon): navigate and inspect — note the returned Interactables
            await page.goto(url, wait_until="domcontentloaded")
            return await mgr.get_content(page)   # or: return await interactables()
            # call 2 (act): use a selector you saw above
        For multi-step work, pass a session_id so the recon'd page persists across
        calls (a one-off page is closed after each call, losing that context).

        Gotchas:
          - console.* does NOT work in-page (the browser's Console API is disabled).
            Use `return`/`print` from Python instead of console.log.
          - Prefer `wait_until="domcontentloaded"` over `"networkidle"` (networkidle
            often times out on pages with long-polling / analytics).
          - Default timeout is 2 × BROWSER_TIMEOUT (min 60s); raise it with `timeout=`.
          - Text alone misses a lot: when you need VISUAL context — page layout, the
            rendered state, an image/chart, confirming a click/scroll worked, or
            seeing what's blocking you (modal, cookie banner, captcha, login wall) —
            take a browser_screenshot (same session_id) rather than guessing from text.

        Sessions: pass a stable session_id to keep the page/cookies alive across
        calls. Keep ids SHORT to save tokens but make them UNIQUE per task — a
        2-4 char topic hint plus a digit works well, e.g. "wm1", "shop2", "log1"
        (a bare word like "walmart" is too collision-prone; a long random hash
        wastes tokens). Reuse that exact id on follow-up calls for the same task.
        Omit session_id for a one-off page that is closed after the call.

        Parallelism: independent browser jobs run AT THE SAME TIME. To do that,
        emit several browser_run calls in a SINGLE turn — one per job, each with
        its OWN unique session_id (e.g. "wm1", "amz1", "ebay1"). They execute
        concurrently in separate browser contexts, so prefer batching independent
        navigations/extractions in one turn instead of going one-at-a-time across
        turns. Only chain calls sequentially when a later step depends on an
        earlier one (then reuse that step's session_id). Use browser_screenshot
        for visual checks, browser_close to free a session.

        Example:
            await page.goto("https://example.com", wait_until="domcontentloaded")
            await page.fill("input[name='q']", "playwright")
            await page.click("button[type='submit']")
            await page.wait_for_load_state("domcontentloaded")
            return await page.title()

        SECURITY: code runs in-process (full Python/host access), gated behind the
        server's API key. Intended for trusted local use.

        Large stdout/string results are previewed with a read_output handle shown
        in their section; call read_output(handle=...) to read the rest.
        Returns: markdown — a status/session line, the page title+URL, the returned
        result, stdout, and the visible interactable elements. ON ERROR the same
        report is returned with status=error plus an **Error** block (traceback with
        the failing code line), a corrective hint, the page URL/title reached so
        far, any stdout printed before the failure, and the Interactables list — so
        you can see where you were and fix the selector/step and retry.
        """
        page = None
        sid = session_id
        one_off = session_id is None
        buf = io.StringIO()
        try:
            # Parse the code up front so progress is driven by the actual steps
            # it performs (one labelled update per top-level statement).
            compiled, step_labels = _instrument_browser_code(code)
            total = len(step_labels) or 1

            async def __report(idx: int):
                if ctx and 0 <= idx < len(step_labels):
                    with contextlib.suppress(Exception):
                        await ctx.report_progress(idx, total, step_labels[idx])

            if ctx:
                await ctx.report_progress(0, total, "Preparing browser page\u2026")
            page, sid = await _ensure_page(session_id)
            mgr = _get_browser_manager()

            async def interactables():
                return await mgr.get_interactables(page)

            exec_globals = {"asyncio": asyncio, "json": json, "re": re}
            ns: dict = {}
            exec(compiled, exec_globals, ns)

            eff_timeout = timeout or max(settings.BROWSER_TIMEOUT * 2, 60)
            with contextlib.redirect_stdout(buf):
                result = await asyncio.wait_for(
                    ns["__browser_main"](page, page.context, mgr, interactables, __report),
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

            # Paginate oversized stdout / string results via read_output.
            holder: dict = {}
            output_store.attach(holder, "stdout", stdout_full, source="browser_run stdout")
            result_is_text = isinstance(result_repr, str)
            if result_is_text and len(result_repr) > settings.OUTPUT_PREVIEW_CHARS:
                output_store.attach(holder, "result", result_repr, source="browser_run result")

            body_parts: list[str] = []
            if result_repr is not None:
                body_parts.append("")
                if result_is_text:
                    if "result" in holder:
                        body_parts += ["**Result (preview):**", "```text", holder["result"], "```"]
                        if "result_hint" in holder:
                            body_parts.append(f"_{holder['result_hint']}_")
                    elif "\n" in result_repr:
                        body_parts += ["**Result:**", "```text", result_repr, "```"]
                    else:
                        body_parts.append(f"**Result:** {result_repr}")
                elif isinstance(result_repr, (int, float, bool)):
                    body_parts.append(f"**Result:** `{result_repr}`")
                else:
                    body_parts += ["**Result:**", "```json", json.dumps(result_repr, indent=2, default=str), "```"]

            if ctx:
                await ctx.report_progress(total, total, "Done")
            return _render_run_report(
                status="success",
                sid=sid,
                title=title,
                url=url,
                body_parts=body_parts,
                stdout_preview=holder.get("stdout", ""),
                stdout_hint=holder.get("stdout_hint"),
                summary=summary,
            )
        except Exception as e:
            logger.error("browser_run error: %s", str(e))
            # Build a recoverable error report: traceback (with the failing code
            # line), a corrective hint, the page state reached so far, any stdout
            # captured before the crash, and the live interactables list.
            error_block = _clean_browser_traceback(e)
            hint = _failure_hint(e)
            url = title = None
            summary = ""
            if page is not None:
                with contextlib.suppress(Exception):
                    url, title = page.url, await page.title()
                with contextlib.suppress(Exception):
                    summary = await _interactables_summary(page)

            holder = {}
            output_store.attach(holder, "stdout", buf.getvalue(), source="browser_run stdout")
            return _render_run_report(
                status="error",
                sid=sid,
                title=title,
                url=url,
                body_parts=[],
                stdout_preview=holder.get("stdout", ""),
                stdout_hint=holder.get("stdout_hint"),
                summary=summary,
                error_block=error_block,
                hint=hint,
            )
        finally:
            if one_off and page is not None:
                with contextlib.suppress(Exception):
                    await _get_browser_manager().close_session(sid)

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
            Field(description="Screenshot the current page of this persistent session (from browser_run)."),
        ] = None,
        ctx: Context | None = None,
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
            info = (
                f"Screenshot saved: file://{screenshot_path}\n"
                f"Session: {sid or 'one-off'}  Full page: {full_page}"
            )
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

    @server.tool()
    async def browser_sessions(ctx: Context | None = None) -> str:
        """List live browser sessions so you can reuse one instead of guessing ids.

        Returns each session's id, current page url+title, open page count, and
        idle seconds. Use this when unsure which session_id is active, or to
        check whether a previous browser_run left a session open.
        """
        try:
            mgr = _get_browser_manager()
            now = time.time()
            sessions = []
            for sid, session in mgr._sessions.items():
                page = session.pages[-1] if session.pages else None
                url = None
                title = None
                if page is not None:
                    with contextlib.suppress(Exception):
                        url = page.url
                    with contextlib.suppress(Exception):
                        title = await page.title()
                sessions.append({
                    "session_id": sid,
                    "current_url": url,
                    "current_title": title,
                    "pages": len(session.pages),
                    "idle_seconds": round(now - session.last_activity, 1),
                })
            return json.dumps({
                "status": "success",
                "count": len(sessions),
                "sessions": sessions,
            })
        except Exception as e:
            logger.error("browser_sessions error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_close(
        session_id: Annotated[str, Field(description="The session id you passed to browser_run.")],
        ctx: Context | None = None,
    ) -> str:
        """Close a browser session and free its pages/cookies.

        Pass the session_id you used with browser_run.
        """
        try:
            if ctx:
                await ctx.report_progress(0, 1, f"Closing session {session_id}\u2026")
            ok = await _get_browser_manager().close_session(session_id)
            if ctx:
                await ctx.report_progress(1, 1, "Done")
            return json.dumps({
                "status": "success" if ok else "error",
                "message": f"Session {session_id} {'closed' if ok else 'not found'}",
            })
        except Exception as e:
            logger.error("browser_close error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    logger.info("Registered browser automation tools (browser_run, browser_screenshot, browser_sessions, browser_close)")
