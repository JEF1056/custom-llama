"""Browser automation tools for MCP server."""

import asyncio
import base64
import json
import logging
import os

from mcp.server import FastMCP
from mcp.types import ImageContent, TextContent

from src.browser.automation import browser_manager
from src.config import settings

logger = logging.getLogger(__name__)


def _get_page_for_session(session_id: str):
    """Get the last page from a session, or None if the session has no pages."""
    session = browser_manager._sessions.get(session_id)
    if session and session.pages:
        return session.pages[-1]
    return None


async def _ensure_page(session_id: str):
    """Get existing page from session or create a new one and store it."""
    page = _get_page_for_session(session_id)
    if page is not None:
        browser_manager._touch_session(session_id)
        return page
    # Create a new page in the session's context
    session = browser_manager._sessions.get(session_id)
    if not session:
        raise RuntimeError(f"Session {session_id} not found. Create one with browser_create_session first.")
    page = await session.context.new_page()
    session.pages.append(page)
    return page


async def _interactables_summary(page) -> str:
    """Extract and format a summary of interactable elements on the page."""
    try:
        elements = await browser_manager.get_interactables(page)
        visible = [e for e in elements if e.get("visible")]
        if not visible:
            return ""
        lines = [
            f"\n=== Interactable elements "
            f"({len(visible)} visible / {len(elements)} total) ==="
        ]
        for el in visible[:40]:
            tag = el.get("tag", "")
            etype = el.get("type", "")
            parts = [el.get("text", ""), el.get("name", ""),
                     el.get("placeholder", "")]
            text = (next((p for p in parts if p), f"<{tag}>"))[:80]
            sel = el.get("selector", "")
            line = f"  [{el['index']}] {etype} {tag}: "
            line += f"\"{text}\" → selector: {sel}"
            lines.append(line)
        if len(visible) > 40:
            lines.append(f"  ... and {len(visible) - 40} more visible elements")
        return "\n".join(lines)
    except Exception:
        return ""


def browser_handler(server: FastMCP) -> None:
    """Register browser automation tools."""

    @server.tool()
    async def browser_create_session() -> str:
        """Create a browser session for multi-step interactions.

        Call first when performing multiple actions on the same page.
        Pass the returned session_id to all subsequent browser tools.
        For one-off actions, use browser_screenshot(url=...) directly.
        """
        try:
            if not browser_manager.is_running:
                await browser_manager.start()

            session_id = await browser_manager.create_session()
            result = {
                "status": "success",
                "session_id": session_id,
                "message": "Browser session created",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser create_session error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_navigate(
        url: str,
        wait_until: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Navigate to a URL.

        Load a page before further actions (click, fill, screenshot). Provide
        session_id to persist the page; without it, the page closes immediately.
        wait_until options: load, domcontentloaded, networkidle, commit.
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                # Use session-based page tracking
                page = await _ensure_page(session_id)
                await page.goto(
                    url,
                    timeout=settings.BROWSER_TIMEOUT * 1000,
                    wait_until=wait_until or "domcontentloaded",
                )
            else:
                # One-off navigation — create page, navigate, close
                context = await browser_manager.get_session(None)
                if not context:
                    raise RuntimeError("Browser not running")
                page = await context.new_page()
                await page.goto(
                    url,
                    timeout=settings.BROWSER_TIMEOUT * 1000,
                    wait_until=wait_until or "domcontentloaded",
                )

            # Get page info
            title = await page.title()
            current_url = page.url

            # Extract interactable elements for the LLM
            summary = await _interactables_summary(page)

            # Close page only for one-off (no session_id)
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "url": current_url,
                "title": title,
                "session_id": session_id or "none",
                "interactables_summary": summary,
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser navigate error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_screenshot(
        url: str | None = None,
        full_page: bool = False,
        session_id: str | None = None,
    ):
        """Screenshot a URL or session page. Returns MCP ImageContent (base64 PNG) + resource URI.

        One-off: provide url=. Session: provide session_id=.
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                # Use session-based page tracking
                page = await _ensure_page(session_id)
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )
            else:
                # One-off screenshot
                context = await browser_manager.get_session(None)
                if not context:
                    raise RuntimeError("Browser not running")
                page = await context.new_page()
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )

            # Take screenshot - returns (bytes, file_path)
            screenshot_bytes, screenshot_path = await browser_manager.screenshot(
                page,
                full_page=full_page,
                session_id=session_id,
            )

            # Extract interactable elements for the LLM
            summary = await _interactables_summary(page)

            # Clean up page only for one-off (no session_id)
            if not session_id:
                await page.close()

            # Build resource URI
            resource_uri = f"file://{screenshot_path}"

            # Return as MCP ImageContent (base64 PNG)
            image_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            image = ImageContent(
                type="image",
                data=image_b64,
                mimeType="image/png",
            )

            info_text = (
                f"Screenshot captured successfully.\n"
                f"  Resource URI: {resource_uri}\n"
                f"  Session: {session_id or 'none'}\n"
                f"  Full page: {full_page}\n"
                f"\n"
                f"The screenshot is embedded above as an MCP image. "
                f"You can also access it later via the resource URI."
            )
            if summary:
                info_text += summary

            return [image, TextContent(type="text", text=info_text)]
        except Exception as e:
            logger.error("Browser screenshot error: %s", str(e))
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    @server.tool()
    async def browser_click(
        selector: str,
        url: str | None = None,
        timeout: int | None = None,
        wait_until: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Click an element. Requires session_id; use url= to navigate first.

        TIP: Call browser_get_interactables first to find the right selector.
        wait_until: navigation trigger after click.
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                page = await _ensure_page(session_id)
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )
            else:
                context = await browser_manager.get_session(None)
                if not context:
                    return json.dumps({"status": "error", "error": "Browser not running"})
                page = await context.new_page()
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )

            # Click the element
            await browser_manager.click(page, selector)

            # Wait for navigation if specified
            if wait_until:
                await page.wait_for_load_state(wait_until)

            # Extract interactable elements for the LLM
            summary = await _interactables_summary(page)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "message": f"Clicked element: {selector}",
                "session_id": session_id or "none",
                "interactables_summary": summary,
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser click error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_fill(
        selector: str,
        value: str,
        url: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Fill an input field. Requires session_id; use url= to navigate first.

        TIP: Call browser_get_interactables first to find the right selector.
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                page = await _ensure_page(session_id)
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )
            else:
                context = await browser_manager.get_session(None)
                if not context:
                    return json.dumps({"status": "error", "error": "Browser not running"})
                page = await context.new_page()
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )

            # Fill the input
            await browser_manager.fill(page, selector, value)

            # Extract interactable elements for the LLM
            summary = await _interactables_summary(page)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "message": f"Filled element: {selector} with {value}",
                "session_id": session_id or "none",
                "interactables_summary": summary,
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser fill error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_evaluate(
        script: str,
        url: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Execute JavaScript on the current page. Requires session_id; use url= to navigate first."""
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                page = await _ensure_page(session_id)
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )
            else:
                context = await browser_manager.get_session(None)
                if not context:
                    return json.dumps({"status": "error", "error": "Browser not running"})
                page = await context.new_page()
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )

            # Execute JavaScript
            result = await browser_manager.evaluate(page, script)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            return json.dumps({
                "status": "success",
                "result": result,
                "session_id": session_id or "none",
            }, indent=2)
        except Exception as e:
            logger.error("Browser evaluate error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_get_text(
        selector: str,
        url: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Get text content of an element. Requires session_id; use url= to navigate first."""
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                page = await _ensure_page(session_id)
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )
            else:
                context = await browser_manager.get_session(None)
                if not context:
                    return json.dumps({"status": "error", "error": "Browser not running"})
                page = await context.new_page()
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )

            # Get text
            text = await browser_manager.get_text(page, selector)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            return json.dumps({
                "status": "success",
                "text": text,
                "session_id": session_id or "none",
            }, indent=2)
        except Exception as e:
            logger.error("Browser get_text error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_get_interactables(
        url: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """List all interactable elements (links, buttons, inputs) with selectors.

        Use BEFORE clicking/filling to find the right selector.
        Requires session_id; use url= to navigate first.
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                page = await _ensure_page(session_id)
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )
            else:
                context = await browser_manager.get_session(None)
                if not context:
                    return json.dumps({"status": "error", "error": "Browser not running"})
                page = await context.new_page()
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )

            # Get interactable elements
            elements = await browser_manager.get_interactables(page)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            # Build a readable summary
            visible = [e for e in elements if e.get("visible")]
            hidden = [e for e in elements if not e.get("visible")]

            lines = [
                f"Found {len(elements)} interactable elements "
                f"({len(visible)} visible, {len(hidden)} hidden):\n",
            ]

            if visible:
                lines.append("=== VISIBLE ELEMENTS ===")
                for el in visible:
                    tag = el.get("tag", "")
                    etype = el.get("type", "")
                    text = el.get("text", "")[:80]
                    sel = el.get("selector", "")
                    name = el.get("name", "")
                    placeholder = el.get("placeholder", "")
                    label = text or name or placeholder or f"<{tag}>"
                    lines.append(f"  [{el['index']}] {etype} {tag}: \"{label}\"")
                    lines.append(f"      selector: {sel}")
                lines.append("")

            if hidden:
                lines.append("=== HIDDEN ELEMENTS ===")
                for el in hidden:
                    tag = el.get("tag", "")
                    etype = el.get("type", "")
                    text = el.get("text", "")[:80]
                    sel = el.get("selector", "")
                    name = el.get("name", "")
                    placeholder = el.get("placeholder", "")
                    label = text or name or placeholder or f"<{tag}>"
                    lines.append(f"  [{el['index']}] {etype} {tag}: \"{label}\"")
                    lines.append(f"      selector: {sel}")

            result = {
                "status": "success",
                "total": len(elements),
                "visible_count": len(visible),
                "hidden_count": len(hidden),
                "elements": elements,
                "summary": "\n".join(lines),
                "session_id": session_id or "none",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser get_interactables error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_get_content(
        url: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Get full page text content. Requires session_id; use url= to navigate first.

        For one-off extraction, use fetch() instead.
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            if session_id:
                page = await _ensure_page(session_id)
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )
            else:
                context = await browser_manager.get_session(None)
                if not context:
                    return json.dumps({"status": "error", "error": "Browser not running"})
                page = await context.new_page()
                if url:
                    await page.goto(
                        url,
                        timeout=settings.BROWSER_TIMEOUT * 1000,
                        wait_until="domcontentloaded",
                    )

            # Get content
            content = await browser_manager.get_content(page)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            return json.dumps({
                "status": "success",
                "content": content,
                "content_length": len(content),
                "session_id": session_id or "none",
            }, indent=2)
        except Exception as e:
            logger.error("Browser get_content error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_monitor(
        url: str | None = None,
        interval: int = 5,
        duration: int = 30,
        path: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Capture periodic screenshots. Requires session_id; use url= to navigate first.

        Returns file:// resource URIs for each screenshot.
        interval: seconds between shots (default 5). duration: total seconds (default 30).
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Use provided path or default screenshot directory
            output_dir = path or browser_manager.screenshot_dir
            os.makedirs(output_dir, exist_ok=True)

            # Calculate number of screenshots
            num_screenshots = duration // interval

            if session_id:
                page = await _ensure_page(session_id)
            else:
                context = await browser_manager.get_session(None)
                if not context:
                    return json.dumps({"status": "error", "error": "Browser not running"})
                page = await context.new_page()

            # Navigate if URL provided
            if url:
                await page.goto(
                    url,
                    timeout=settings.BROWSER_TIMEOUT * 1000,
                    wait_until="domcontentloaded",
                )

            screenshot_uris = []

            for i in range(num_screenshots):
                screenshot_bytes = await page.screenshot(full_page=False)
                screenshot_path = os.path.join(
                    output_dir, f"monitor_{session_id or 'oneoff'}_{i}.png"
                )
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot_bytes)
                resource_uri = f"file://{screenshot_path}"
                screenshot_uris.append(resource_uri)
                logger.info("Monitor screenshot %d saved to %s", i + 1, screenshot_path)

                if i < num_screenshots - 1:
                    await asyncio.sleep(interval)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "message": f"Captured {len(screenshot_uris)} screenshots",
                "screenshot_uris": screenshot_uris,
                "session_id": session_id or "none",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser monitor error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_close(
        session_id: str | None = None,
    ) -> str:
        """Close a browser session. session_id=None closes the default context."""
        try:
            success = await browser_manager.close_session(session_id)
            if success:
                result = {
                    "status": "success",
                    "message": f"Session {session_id or 'default'} closed",
                }
            else:
                result = {
                    "status": "error",
                    "error": f"Session {session_id or 'default'} not found",
                }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser close error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_list_sessions() -> str:
        """List active browser sessions."""
        try:
            sessions = browser_manager.active_sessions
            result = {
                "status": "success",
                "sessions": sessions,
                "total": len(sessions),
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser list_sessions error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    logger.info("Registered browser automation tools")
