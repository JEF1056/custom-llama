"""Browser automation tools for MCP server."""

import asyncio
import json
import logging
import os
from typing import Any

from mcp.server import FastMCP

from src.browser.automation import browser_manager

logger = logging.getLogger(__name__)


def browser_handler(server: FastMCP) -> None:
    """Register all browser automation tools with the MCP server.

    Args:
        server: The MCP server instance.
    """

    @server.tool()
    async def browser_navigate(
        url: str,
        wait_until: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Navigate to a URL.

        Args:
            url: The URL to navigate to
            wait_until: When to consider navigation completed (load, domcontentloaded, networkidle, commit)
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with page title, URL, and status
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Navigate to the URL
            page = await browser_manager.goto(
                url,
                wait_until=wait_until or "networkidle",
                session_id=session_id,
            )

            # Get page info
            title = await page.title()
            current_url = page.url

            # Clean up page
            await page.close()

            result = {
                "status": "success",
                "url": current_url,
                "title": title,
                "session_id": session_id or "default",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser navigate error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_screenshot(
        full_page: bool = False,
        path: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Take a screenshot of the current page.

        Args:
            full_page: Whether to capture the full page
            path: The file path to save the screenshot. If None, saves to screenshot_dir.
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with screenshot path
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Get the session context
            context = await browser_manager.get_session(session_id)
            if not context:
                return json.dumps({"status": "error", "error": "Browser not running"})

            # Create a new page for the screenshot
            page = await context.new_page()

            # Take screenshot
            screenshot_path = await browser_manager.screenshot(
                page,
                path=path,
                full_page=full_page,
                session_id=session_id,
            )

            # Clean up page
            await page.close()

            result = {
                "status": "success",
                "screenshot_path": screenshot_path,
                "session_id": session_id or "default",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser screenshot error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_click(
        selector: str,
        timeout: int | None = None,
        wait_until: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Click an element.

        Args:
            selector: The CSS selector for the element
            timeout: Timeout in seconds. Defaults to settings.BROWSER_TIMEOUT.
            wait_until: When to consider navigation completed after click
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with success/failure status
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Get the session context
            context = await browser_manager.get_session(session_id)
            if not context:
                return json.dumps({"status": "error", "error": "Browser not running"})

            # Create a new page
            page = await context.new_page()

            # Click the element
            await browser_manager.click(page, selector)

            # Wait for navigation if specified
            if wait_until:
                await page.wait_for_load_state(wait_until)

            # Clean up page
            await page.close()

            result = {
                "status": "success",
                "message": f"Clicked element: {selector}",
                "session_id": session_id or "default",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser click error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_fill(
        selector: str,
        value: str,
        session_id: str | None = None,
    ) -> str:
        """Fill an input field.

        Args:
            selector: The CSS selector for the input
            value: The value to fill
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with success/failure status
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Get the session context
            context = await browser_manager.get_session(session_id)
            if not context:
                return json.dumps({"status": "error", "error": "Browser not running"})

            # Create a new page
            page = await context.new_page()

            # Fill the input
            await browser_manager.fill(page, selector, value)

            # Clean up page
            await page.close()

            result = {
                "status": "success",
                "message": f"Filled element: {selector} with {value}",
                "session_id": session_id or "default",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser fill error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_evaluate(
        script: str,
        session_id: str | None = None,
    ) -> str:
        """Execute JavaScript on the page.

        Args:
            script: The JavaScript code to execute
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with the result of the JavaScript execution
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Get the session context
            context = await browser_manager.get_session(session_id)
            if not context:
                return json.dumps({"status": "error", "error": "Browser not running"})

            # Create a new page
            page = await context.new_page()

            # Execute JavaScript
            result = await browser_manager.evaluate(page, script)

            # Clean up page
            await page.close()

            return json.dumps({
                "status": "success",
                "result": result,
                "session_id": session_id or "default",
            }, indent=2)
        except Exception as e:
            logger.error("Browser evaluate error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_get_text(
        selector: str,
        session_id: str | None = None,
    ) -> str:
        """Get text content of an element.

        Args:
            selector: The CSS selector for the element
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with the text content
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Get the session context
            context = await browser_manager.get_session(session_id)
            if not context:
                return json.dumps({"status": "error", "error": "Browser not running"})

            # Create a new page
            page = await context.new_page()

            # Get text
            text = await browser_manager.get_text(page, selector)

            # Clean up page
            await page.close()

            return json.dumps({
                "status": "success",
                "text": text,
                "session_id": session_id or "default",
            }, indent=2)
        except Exception as e:
            logger.error("Browser get_text error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_get_content(
        session_id: str | None = None,
    ) -> str:
        """Get the page content (text extraction).

        Args:
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with page text content
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Get the session context
            context = await browser_manager.get_session(session_id)
            if not context:
                return json.dumps({"status": "error", "error": "Browser not running"})

            # Create a new page
            page = await context.new_page()

            # Get content
            content = await browser_manager.get_content(page)

            # Clean up page
            await page.close()

            return json.dumps({
                "status": "success",
                "content": content,
                "content_length": len(content),
                "session_id": session_id or "default",
            }, indent=2)
        except Exception as e:
            logger.error("Browser get_content error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_monitor(
        interval: int = 5,
        duration: int = 30,
        path: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Periodic screenshot monitoring.

        Captures screenshots at regular intervals for a specified duration.

        Args:
            interval: Seconds between screenshots (default: 5)
            duration: Total seconds to monitor (default: 30)
            path: Output directory for screenshots. If None, uses screenshot_dir.
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            JSON string with list of screenshot paths
        """
        try:
            # Start browser if not running
            if not browser_manager.is_running:
                await browser_manager.start()

            # Get the session context
            context = await browser_manager.get_session(session_id)
            if not context:
                return json.dumps({"status": "error", "error": "Browser not running"})

            # Use provided path or default screenshot directory
            output_dir = path or browser_manager.screenshot_dir
            os.makedirs(output_dir, exist_ok=True)

            # Calculate number of screenshots
            num_screenshots = duration // interval

            screenshot_paths = []
            page = await context.new_page()

            for i in range(num_screenshots):
                screenshot_path = await browser_manager.screenshot(
                    page,
                    path=os.path.join(output_dir, f"monitor_{session_id or 'default'}_{i}.png"),
                    session_id=session_id,
                )
                screenshot_paths.append(screenshot_path)
                logger.info("Monitor screenshot %d saved to %s", i + 1, screenshot_path)

                if i < num_screenshots - 1:
                    await asyncio.sleep(interval)

            # Clean up page
            await page.close()

            result = {
                "status": "success",
                "message": f"Captured {len(screenshot_paths)} screenshots",
                "screenshot_paths": screenshot_paths,
                "session_id": session_id or "default",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error("Browser monitor error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)})

    @server.tool()
    async def browser_close(
        session_id: str | None = None,
    ) -> str:
        """Close the browser session.

        Args:
            session_id: The session ID to close. If None, closes the default context.

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
        """List active browser sessions.

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
