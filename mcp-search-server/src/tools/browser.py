"""Browser automation tools for MCP server."""

import asyncio
import json
import logging
import os
from typing import Any

from mcp.server import FastMCP
from mcp.server.fastmcp import Image

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
        return page
    # Create a new page in the session's context
    session = browser_manager._sessions.get(session_id)
    if not session:
        raise RuntimeError(f"Session {session_id} not found. Create one with browser_create_session first.")
    page = await session.context.new_page()
    session.pages.append(page)
    return page


def browser_handler(server: FastMCP) -> None:
    """Register all browser automation tools with the MCP server.

    Args:
        server: The MCP server instance.
    """

    @server.tool()
    async def browser_create_session() -> str:
        """Create a new browser session for multi-step interactions.

        Call this first when you need to perform multiple browser actions (navigate,
        click, fill, screenshot, etc.) on the same page. The returned session_id must
        be passed to all subsequent browser tools to maintain page state.

        For one-off actions (e.g., just a screenshot of a URL), use browser_screenshot(url=...) directly.

        Returns:
            JSON string containing the session_id.
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
        """Navigate to a URL in a browser session.

        Use this to load a page before performing further actions (click, fill, screenshot,
        etc.). Requires a session_id from browser_create_session() to persist the page.
        Without session_id, the page is navigated and immediately closed.

        Args:
            url: The URL to navigate to
            wait_until: When to consider navigation completed (load, domcontentloaded, networkidle, commit)
            session_id: Session ID from browser_create_session(). Required to keep the page alive.

        Returns:
            JSON string with page title, URL, and status
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

            # Close page only for one-off (no session_id)
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "url": current_url,
                "title": title,
                "session_id": session_id or "none",
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
    ) -> Image:
        """Take a screenshot of a URL or the current page in a session.

        Two usage modes:
        1. One-off: provide url=... to navigate and screenshot in a single call.
        2. Session: provide session_id=... to screenshot the current page from an active session.

        Args:
            url: URL to navigate to and screenshot (required for one-off mode)
            full_page: Whether to capture the full page
            session_id: Session ID from browser_create_session() (required for session mode)

        Returns:
            Image object containing the screenshot PNG data.
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

            # Take screenshot - returns raw PNG bytes
            screenshot_bytes = await browser_manager.screenshot(
                page,
                full_page=full_page,
            )

            # Clean up page only for one-off (no session_id)
            if not session_id:
                await page.close()

            return Image(data=screenshot_bytes, format="png")
        except Exception as e:
            logger.error("Browser screenshot error: %s", str(e))
            raise e

    @server.tool()
    async def browser_click(
        selector: str,
        url: str | None = None,
        timeout: int | None = None,
        wait_until: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Click an element on the current page in a session.

        Requires a session_id from browser_create_session(). The page must already be
        loaded (via browser_navigate or browser_screenshot with url=). Use url= to
        navigate to a new page before clicking.

        Args:
            selector: CSS selector for the element to click
            url: Optional URL to navigate to before clicking
            timeout: Timeout in seconds. Defaults to settings.BROWSER_TIMEOUT.
            wait_until: When to consider navigation completed after click
            session_id: Session ID from browser_create_session(). Required.

        Returns:
            JSON string with success/failure status
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

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "message": f"Clicked element: {selector}",
                "session_id": session_id or "none",
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
        """Fill an input field on the current page in a session.

        Requires a session_id from browser_create_session(). The page must already be
        loaded. Use url= to navigate to a new page before filling.

        Args:
            selector: CSS selector for the input element
            value: The value to fill
            url: Optional URL to navigate to before filling
            session_id: Session ID from browser_create_session(). Required.

        Returns:
            JSON string with success/failure status
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

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "message": f"Filled element: {selector} with {value}",
                "session_id": session_id or "none",
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
        """Execute JavaScript on the current page in a session.

        Requires a session_id from browser_create_session(). The page must already be
        loaded. Use url= to navigate to a new page first.

        Args:
            script: JavaScript code to execute
            url: Optional URL to navigate to before evaluating
            session_id: Session ID from browser_create_session(). Required.

        Returns:
            JSON string with the result of the JavaScript execution
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
        """Get text content of an element on the current page in a session.

        Requires a session_id from browser_create_session(). The page must already be
        loaded. Use url= to navigate to a new page first. Use browser_screenshot(url=...)
        for a one-off screenshot instead.

        Args:
            selector: CSS selector for the element
            url: Optional URL to navigate to before getting text
            session_id: Session ID from browser_create_session(). Required.

        Returns:
            JSON string with the text content
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
    async def browser_get_content(
        url: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Get the full page content of the current page in a session.

        Requires a session_id from browser_create_session(). The page must already be
        loaded. Use url= to navigate to a new page first.For one-off content extraction,
        use the fetch() tool instead.

        Args:
            url: Optional URL to navigate to before getting content
            session_id: Session ID from browser_create_session(). Required.

        Returns:
            JSON string with page text content
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
        """Periodic screenshot monitoring of a page.

        Captures screenshots at regular intervals for a specified duration. Requires a
        session_id from browser_create_session(). Use url= to navigate to a page first.

        Args:
            url: Optional URL to navigate to before monitoring
            interval: Seconds between screenshots (default: 5)
            duration: Total seconds to monitor (default: 30)
            path: Output directory for screenshots. If None, uses screenshot_dir.
            session_id: Session ID from browser_create_session(). Required.

        Returns:
            JSON string with list of screenshot paths
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

            screenshot_paths = []

            for i in range(num_screenshots):
                screenshot_bytes = await page.screenshot(full_page=False)
                screenshot_path = os.path.join(
                    output_dir, f"monitor_{session_id or 'oneoff'}_{i}.png"
                )
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot_bytes)
                screenshot_paths.append(screenshot_path)
                logger.info("Monitor screenshot %d saved to %s", i + 1, screenshot_path)

                if i < num_screenshots - 1:
                    await asyncio.sleep(interval)

            # Clean up page only for one-off
            if not session_id:
                await page.close()

            result = {
                "status": "success",
                "message": f"Captured {len(screenshot_paths)} screenshots",
                "screenshot_paths": screenshot_paths,
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
        """Close a browser session.

        Use this when done with a session created by browser_create_session().
        Frees resources and closes the session's context.

        Args:
            session_id: Session ID to close. If None, closes the default context.

        Returns:
            JSON string with success/failure status
        """
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
        """List all active browser sessions.

        Returns the session IDs of all sessions created by browser_create_session().

        Returns:
            JSON string with list of session IDs
        """
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
