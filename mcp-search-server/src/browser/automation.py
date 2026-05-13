"""Browser automation using Playwright."""

import asyncio
import logging
import random
from typing import Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from src.config import settings

logger = logging.getLogger(__name__)

# Common user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv/121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv/121.0) Gecko/20100101 Firefox/121.0",
]


class BrowserManager:
    """Manages Playwright browser instances with anti-detection features."""

    def __init__(self):
        """Initialize the browser manager."""
        self._playwright = None
        self._browser = None
        self._context = None

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
        # Rotate user agent for anti-detection
        user_agent = random.choice(USER_AGENTS)
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=user_agent,
            ignore_https_errors=True,
            java_enabled=False,
        )
        # Set extra HTTP headers to avoid detection
        await self._context.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        })
        logger.info("Browser started with user agent: %s", user_agent[:50])

    async def stop(self) -> None:
        """Stop the browser."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser stopped")

    async def goto(self, url: str, timeout: int | None = None) -> Page:
        """Navigate to a URL with anti-detection measures.

        Args:
            url: The URL to navigate to.
            timeout: Timeout in seconds. Defaults to settings.BROWSER_TIMEOUT.

        Returns:
            The page object.
        """
        page = await self._context.new_page()
        await page.goto(
            url,
            timeout=(timeout or settings.BROWSER_TIMEOUT) * 1000,
            wait_until="networkidle",
        )
        # Add random delay to avoid bot detection
        await self._add_delay()
        return page

    async def _add_delay(self) -> None:
        """Add a random delay to simulate human behavior."""
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)

    async def screenshot(self, page: Page, path: str) -> str:
        """Take a screenshot of the page.

        Args:
            page: The page to screenshot.
            path: The file path to save the screenshot.

        Returns:
            The path where the screenshot was saved.
        """
        await page.screenshot(path=path, full_page=True)
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
        return self._browser is not None and self._context is not None


# Global browser manager instance
browser_manager = BrowserManager()
