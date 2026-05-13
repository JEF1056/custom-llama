"""Browser automation using Playwright with session management."""

import asyncio
import logging
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from src.config import settings

logger = logging.getLogger(__name__)

# Common user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome:120.0.0.0 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64: x64) AppleWebKit/537.36 (KHTML: like Gecko) Chrome/119.0.0.0 Safari/537.3",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML: like Gecko) Chrome/120.0.0.0 Safari:537.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 19.15; rv/121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0: Win64: x64; rv/121.0) Gecko/2010101 Firefox/121.0",
]


@dataclass
class BrowserSession:
    """Represents a browser session with its own context and pages."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: BrowserContext | None = field(default=None)
    pages: list[Page] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: "")

    @property
    def is_active(self) -> bool:
        """Check if the session is active."""
        return self.context is not None

    async def close(self) -> None:
        """Close all pages and the context."""
        for page in self.pages:
            try:
                await page.close()
            except Exception:
                pass
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
        self.context = None
        self.pages = []
        logger.info("Session %s closed", self.session_id)


class BrowserManager:
    """Manages Playwright browser instances with anti-detection features and session support."""

    def __init__(self, screenshot_dir: str | None = None):
        """Initialize the browser manager.

        Args:
            screenshot_dir: Directory to save screenshots. Defaults to /app/screenshots.
        """
        self._playwright = None
        self._browser: Browser | None = None
        self._default_context: BrowserContext | None = None
        self._sessions: dict[str, BrowserSession] = {}
        self._screenshot_dir = screenshot_dir or settings.SCREENSHOT_DIR
        # Ensure screenshot directory exists
        os.makedirs(self._screenshot_dir, exist_ok=True)

    async def start(self) -> None:
        """Start the browser with anti-detection settings."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        # Create default context
        user_agent = random.choice(USER_AGENTS)
        self._default_context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=user_agent,
            ignore_https_errors=True,
            java_enabled=False,
        )
        await self._default_context.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        })
        logger.info("Browser started with user agent: %s", user_agent[:50])

    async def stop(self) -> None:
        """Stop the browser and close all sessions."""
        # Close all sessions
        for session_id, session in list(self._sessions.items()):
            await session.close()
            logger.info("Closed session %s during shutdown", session_id)
        self._sessions.clear()

        # Close default context
        if self._default_context:
            await self._default_context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser stopped")

    async def create_session(self) -> str:
        """Create a new browser session.

        Returns:
            The session ID.
        """
        if not self.is_running:
            await self.start()

        session = BrowserSession()
        session.context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=random.choice(USER_AGENTS),
            ignore_https_errors=True,
            java_enabled=False,
        )
        await session.context.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif",
        })
        self._sessions[session.session_id] = session
        logger.info("Created new session: %s", session.session_id)
        return session.session_id

    async def close_session(self, session_id: str | None = None) -> bool:
        """Close a browser session.

        Args:
            session_id: The session ID to close. If None, closes the default context.

        Returns:
            True if session was closed, False if session not found.
        """
        if session_id:
            session = self._sessions.get(session_id)
            if session:
                await session.close()
                del self._sessions[session_id]
                logger.info("Closed session: %s", session_id)
                return True
            return False
        else:
            # Close default context
            if self._default_context:
                await self._default_context.close()
                self._default_context = None
                logger.info("Closed default context")
                return True
            return False

    async def get_session(self, session_id: str | None = None) -> BrowserContext | None:
        """Get a browser context by session ID.

        Args:
            session_id: The session ID. If None, returns the default context.

        Returns:
            The browser context, or None if not found.
        """
        if session_id:
            session = self._sessions.get(session_id)
            if session and session.is_active:
                return session.context
            return None
        return self._default_context

    async def goto(
        self,
        url: str,
        timeout: int | None = None,
        wait_until: str = "networkidle",
        session_id: str | None = None,
    ) -> Page:
        """Navigate to a URL with anti-detection measures.

        Args:
            url: The URL to navigate to.
            timeout: Timeout in seconds. Defaults to settings.BROWSER_TIMEOUT.
            wait_until: When to consider navigation completed.
            session_id: The session ID to use. If None, uses the default context.

        Returns:
            The page object.
        """
        context = await self.get_session(session_id)
        if not context:
            raise RuntimeError("Browser not running")

        page = await context.new_page()
        await page.goto(
            url,
            timeout=(timeout or settings.BROWSER_TIMEOUT) * 1000,
            wait_until=wait_until,
        )
        # Add random delay to avoid bot detection
        await self._add_delay()
        return page

    async def _add_delay(self) -> None:
        """Add a random delay to simulate human behavior."""
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)

    async def screenshot(
        self,
        page: Page,
        path: str | None = None,
        full_page: bool = False,
        session_id: str | None = None,
    ) -> str:
        """Take a screenshot of the page.

        Args:
            page: The page to screenshot.
            path: The file path to save the screenshot. If None, saves to screenshot_dir.
            full_page: Whether to capture the full page.
            session_id: The session ID. Used for default path generation.

        Returns:
            The path where the screenshot was saved.
        """
        if path is None:
            timestamp = asyncio.get_event_loop().time()
            session_prefix = session_id[:8] if session_id else "default"
            path = os.path.join(
                self._screenshot_dir,
                f"screenshot_{session_prefix}_{timestamp:.0f}.png",
            )

        os.makedirs(os.path.dirname(path) or self._screenshot_dir, exist_ok=True)
        await page.screenshot(path=path, full_page=full_page)
        logger.info("Screenshot saved to %s", path)
        return path

    async def get_content(self, page: Page) -> str:
        """Get the page content.

        Args:
            page: The page to get content from.

        Returns:
            The HTML content of the page.
        """
        content = await page.content()
        logger.info("Page content retrieved (%d bytes)", len(content))
        return content

    async def get_text(self, page: Page, selector: str) -> str:
        """Get text content of an element.

        Args:
            page: The page containing the element.
            selector: The CSS selector for the element.

        Returns:
            The text content of the element.
        """
        element = await page.wait_for_selector(selector, timeout=5000)
        if element:
            text = await element.inner_text()
            logger.info("Element text retrieved: %s", text[:100])
            return text
        return ""

    async def click(self, page: Page, selector: str) -> None:
        """Click an element.

        Args:
            page: The page containing the element.
            selector: The CSS selector for the element.
        """
        await page.click(selector, timeout=5000)
        logger.info("Clicked element: %s", selector)

    async def fill(self, page: Page, selector: str, value: str) -> None:
        """Fill an input field.

        Args:
            page: The page containing the input.
            selector: The CSS selector for the input.
            value: The value to fill.
        """
        await page.fill(selector, value, timeout=5000)
        logger.info("Filled element: %s with %s", selector, value)

    async def evaluate(self, page: Page, script: str) -> Any:
        """Execute JavaScript on the page.

        Args:
            page: The page to execute JavaScript on.
            script: The JavaScript code to execute.

        Returns:
            The result of the JavaScript execution.
        """
        result = await page.evaluate(script)
        logger.info("Executed JavaScript: %s", script[:50])
        return result

    async def execute_with_anti_detection(self, page: Page, action: str, **kwargs) -> Any:
        """Execute an action with anti-detection measures.

        Args:
            page: The page to execute the action on.
            action: The action to perform (click, fill, etc.).
            **kwargs: Additional arguments for the action.

        Returns:
            The result of the action.
        """
        # Add random delay before action
        await self._add_delay()

        # Execute the action
        if action == "click":
            selector = kwargs.get("selector")
            if selector:
                await self.click(page, selector)
        elif action == "fill":
            selector = kwargs.get("selector")
            value = kwargs.get("value")
            if selector and value:
                await self.fill(page, selector, value)

    @property
    def is_running(self) -> bool:
        """Check if the browser is running.

        Returns:
            True if the browser is running, False otherwise.
        """
        return self._browser is not None and self._default_context is not None

    @property
    def active_sessions(self) -> list[str]:
        """Get list of active session IDs.

        Returns:
            List of active session IDs.
        """
        return [sid for sid, session in self._sessions.items() if session.is_active]

    @property
    def screenshot_dir(self) -> str:
        """Get the screenshot directory.

        Returns:
            The screenshot directory path.
        """
        return self._screenshot_dir

    @screenshot_dir.setter
    def screenshot_dir(self, value: str) -> None:
        """Set the screenshot directory.

        Args:
            value: The new screenshot directory path.
        """
        self._screenshot_dir = value
        os.makedirs(value, exist_ok=True)


# Global browser manager instance
browser_manager = BrowserManager()
