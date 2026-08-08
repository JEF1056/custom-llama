"""Browser automation with session management.

Anti-detection strategy (state of the art, 2026):

The strongest defence against modern anti-bot systems (Cloudflare, DataDome,
Akamai, Kasada, Fingerprint.com) is NOT to inject a fake fingerprint — JS
overrides are themselves detectable (``__pwInitScripts`` leak, main-world
execution, ``Object.defineProperty`` own-property tells, Proxy traps). Instead
we run a *genuine* Google Chrome whose fingerprint is real and internally
consistent, driven by a CDP-patched automation driver that closes the protocol
leaks vanilla Playwright exposes.

Layers, in order of importance:
  1. **patchright** (drop-in patched Playwright): closes the ``Runtime.enable``
     CDP leak — the #1 signal used by Cloudflare/DataDome — and runs all
     ``evaluate`` calls in an isolated world (no ``mainWorldExecution`` /
     ``__pwInitScripts`` leak), patches ``Console.enable`` and the
     command-flag tells, and supports closed shadow roots.
  2. **Real Google Chrome** via ``channel="chrome"`` — not bundled Chromium /
     Chrome-for-Testing, which advertise a "HeadlessChrome"/Chromium brand that
     is an instant red flag in ``navigator.userAgentData``.
  3. **Headful under Xvfb** (``BROWSER_HEADLESS=false``) — a real windowed
     browser has genuine ``window.outer*`` dimensions, GPU, and a normal event
     loop; headless is trivially fingerprinted.
  4. **No fingerprint injection** — no custom user-agent, no spoofed headers,
     no ``add_init_script``, no stealth plugin. The real Chrome identity is the
     best identity, and it stays consistent with the host's IP/timezone.
  5. **Trusted input only** — navigation/click/fill use the real CDP input
     pipeline (``isTrusted === true``); we never dispatch synthetic JS events,
     which carry ``isTrusted === false`` and are flagged by behavioural checks.
"""

import asyncio
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# patchright is a drop-in, CDP-patched replacement for Playwright. Importing it
# under the same names keeps the rest of the module identical to Playwright.
from patchright.async_api import async_playwright, Browser, BrowserContext, Page
import html2text

from src.config import settings

logger = logging.getLogger(__name__)

# Chromium flags. Deliberately minimal: patchright already strips the
# automation tells (removes --enable-automation, adds the right
# AutomationControlled handling), so we only add what a containerised Chrome
# genuinely needs plus a real desktop window size for headful mode. We do NOT
# pass fingerprint-affecting or "stealth" flags (e.g. --disable-web-security,
# --disable-gpu) — those break real behaviour and are themselves detectable.
LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--ozone-platform=x11",
    "--window-size=1920,1080",
]


@dataclass
class BrowserSession:
    """Represents a browser session with its own context and pages."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: BrowserContext | None = field(default=None)
    pages: list[Page] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)

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
    """Manages multiple Playwright browser instances with anti-detection features and session support.
    
    Uses a pool of Browser instances to enable true parallel browser automation.
    An asyncio.Semaphore limits concurrent browsers to MAX_CONCURRENT_BROWSERS.
    Each session is assigned to a browser from the pool, allowing independent
    CDP streams and true parallelization of operations.
    """

    MAX_CONCURRENT_BROWSERS = 5  # Concurrent Browser instances; tune based on RAM/CPU

    def __init__(self, screenshot_dir: str | None = None):
        """Initialize the browser manager with a browser pool.

        Args:
            screenshot_dir: Directory to save screenshots. Defaults to /app/mcp-files/screenshots.
        """
        self._playwright = None
        self._browser_pool: list[Browser] = []  # Pool of Browser instances
        self._browser_semaphore: asyncio.Semaphore | None = None  # Limits concurrent browsers
        self._sessions: dict[str, BrowserSession] = {}
        self._session_to_browser: dict[str, int] = {}  # Maps session_id to pool index
        self._screenshot_dir = screenshot_dir or settings.SCREENSHOT_DIR
        self._cleanup_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()  # Prevents concurrent start() calls from racing
        # Ensure screenshot directory exists
        os.makedirs(self._screenshot_dir, exist_ok=True)

    async def start(self) -> None:
        """Launch multiple Chrome browsers via the CDP-patched patchright driver.

        Creates MAX_CONCURRENT_BROWSERS instances to enable true parallelization.
        Uses ``channel="chrome"`` so each browser is genuine Google Chrome (not
        bundled Chromium / Chrome-for-Testing, which expose a Chromium brand
        that flags automation). Falls back to bundled Chromium if a system
        Chrome is unavailable so the server still starts in minimal envs.
        """
        async with self._start_lock:
            if self.is_running:
                return  # Another concurrent caller already started the pool
            await self._start_unlocked()

    async def _start_unlocked(self) -> None:
        """Internal: launch the browser pool. Must be called with _start_lock held."""
        self._playwright = await async_playwright().start()
        self._browser_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_BROWSERS)
        launch_kwargs: dict[str, Any] = {
            "headless": settings.BROWSER_HEADLESS,
            "args": LAUNCH_ARGS,
        }
        
        # Launch multiple browser instances
        channel = None
        for i in range(self.MAX_CONCURRENT_BROWSERS):
            try:
                browser = await self._playwright.chromium.launch(
                    channel="chrome", **launch_kwargs
                )
                channel = "chrome"
            except Exception:
                if i == 0:
                    logger.warning(
                        "Google Chrome (channel=chrome) unavailable; falling back to bundled Chromium. "
                        "Install real Chrome for the strongest anti-detection."
                    )
                browser = await self._playwright.chromium.launch(**launch_kwargs)
                channel = "chromium"
            self._browser_pool.append(browser)
            logger.info("Browser instance %d/%d launched", i + 1, self.MAX_CONCURRENT_BROWSERS)

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            "Browser pool started (%d instances, channel=%s, headless=%s)",
            len(self._browser_pool),
            channel,
            settings.BROWSER_HEADLESS,
        )

    async def _create_hardened_context(self, browser: Browser | None = None) -> BrowserContext:
        """Create a browser context that presents a genuine Chrome fingerprint.

        Deliberately injects nothing: no spoofed user-agent, locale, timezone,
        headers, init script, or stealth shim. patchright + real Chrome already
        present a consistent, IP-coherent identity, and any JS override would
        only add a detectable inconsistency. ``no_viewport=True`` makes the
        context use the real OS window size (1920x1080 here) instead of the
        emulated default viewport that fingerprinters flag.

        Args:
            browser: The Browser instance to create context for. If None, uses first from pool.

        Returns:
            A new BrowserContext backed by the real Chrome fingerprint.
        """
        if browser is None and self._browser_pool:
            browser = self._browser_pool[0]
        if browser is None:
            raise RuntimeError("No browsers in pool")
        context = await browser.new_context(
            no_viewport=True,
            ignore_https_errors=True,
        )
        logger.info("Created context (real Chrome fingerprint, no_viewport)")
        return context

    async def managed_page(self, url: str, session_id: str | None = None):
        """Async context manager for temporary pages that auto-cleanup on exit.

        Usage:
            async with browser_manager.managed_page(url) as page:
                content = await browser_manager.get_content(page)
                # page is automatically closed when exiting this block
        
        Args:
            url: The URL to navigate to.
            session_id: Optional session ID. If None, uses ephemeral session.

        Yields:
            The Page object.
        """
        class ManagedPageContext:
            def __init__(self, mgr, url, session_id):
                self.mgr = mgr
                self.url = url
                self.session_id = session_id
                self.page = None

            async def __aenter__(self):
                self.page = await self.mgr.goto(self.url, session_id=self.session_id)
                return self.page

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.page:
                    try:
                        await self.page.close()
                    except Exception as e:
                        logger.warning("Error closing managed page: %s", str(e))
                return False

        return ManagedPageContext(self, url, session_id)

    async def stop(self) -> None:
        """Stop the browser pool and close all sessions."""
        # Close all sessions
        for session_id, session in list(self._sessions.items()):
            await session.close()
            logger.info("Closed session %s during shutdown", session_id)
        self._sessions.clear()
        self._session_to_browser.clear()

        # Close all browsers in the pool
        for i, browser in enumerate(self._browser_pool):
            try:
                await browser.close()
                logger.info("Closed browser instance %d/%d", i + 1, len(self._browser_pool))
            except Exception as e:
                logger.warning("Error closing browser instance %d: %s", i + 1, e)
        self._browser_pool.clear()
        
        if self._playwright:
            await self._playwright.stop()
        # Cancel background cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("Browser pool stopped")

    async def create_session(self, session_id: str | None = None) -> str:
        """Create a new browser session with a context from a browser in the pool.

        Args:
            session_id: Optional explicit id for the session. If None, a uuid is
                generated. Lets callers reuse a stable, human-chosen name.

        Returns:
            The session ID.
        """
        if not self.is_running:
            await self.start()

        # Select browser from pool using round-robin
        browser_idx = len(self._sessions) % len(self._browser_pool)
        browser = self._browser_pool[browser_idx]

        session = BrowserSession() if session_id is None else BrowserSession(session_id=session_id)
        session.context = await self._create_hardened_context(browser=browser)
        self._sessions[session.session_id] = session
        self._session_to_browser[session.session_id] = browser_idx
        logger.info("Created new session: %s (browser pool index=%d)", session.session_id, browser_idx)
        return session.session_id

    async def close_session(self, session_id: str | None = None) -> bool:
        """Close a browser session.

        Args:
            session_id: The session ID to close. If None, closes all sessions.

        Returns:
            True if session was closed, False if session not found.
        """
        if session_id:
            session = self._sessions.get(session_id)
            if session:
                await session.close()
                del self._sessions[session_id]
                self._session_to_browser.pop(session_id, None)
                logger.info("Closed session: %s", session_id)
                return True
            return False
        else:
            # Close all sessions
            for sid in list(self._sessions.keys()):
                await self.close_session(sid)
            return True

    def _touch_session(self, session_id: str) -> None:
        """Update last_activity timestamp for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.last_activity = time.time()

    async def expire_idle_sessions(self) -> None:
        """Close sessions that have been idle longer than the configured timeout."""
        timeout = settings.SESSION_IDLE_TIMEOUT
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_active and (now - session.last_activity) > timeout
        ]
        for sid in expired:
            logger.info("Expiring idle session %s (idle %.1fs, timeout %ss)", sid, now - self._sessions[sid].last_activity, timeout)
            await self.close_session(sid)
        if expired:
            logger.info("Expired %d idle session(s)", len(expired))

    async def _cleanup_loop(self) -> None:
        """Background task that periodically expires idle sessions."""
        interval = min(settings.SESSION_IDLE_TIMEOUT // 4, 60)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.expire_idle_sessions()
            except Exception:
                logger.exception("Error in session cleanup loop")

    async def get_session(self, session_id: str | None = None) -> BrowserContext | None:
        """Get a browser context by session ID.

        Args:
            session_id: The session ID. If None, creates a new ephemeral session.

        Returns:
            The browser context, or None if not found.
        """
        if session_id:
            session = self._sessions.get(session_id)
            if session and session.is_active:
                self._touch_session(session_id)
                return session.context
            return None
        # No session_id: create ephemeral session
        sid = await self.create_session()
        return self._sessions[sid].context

    async def goto(
        self,
        url: str,
        timeout: int | None = None,
        wait_until: str = "domcontentloaded",
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
        # A couple of real (trusted) mouse moves — synthetic dispatched events
        # have isTrusted=false and are ignored by behavioural detectors.
        try:
            for _ in range(random.randint(2, 4)):
                await page.mouse.move(
                    random.randint(0, 1280),
                    random.randint(0, 720),
                    steps=random.randint(3, 8),
                )
        except Exception:
            pass
        # Simulate a scroll down and back up to avoid "always at top" fingerprint
        await page.evaluate(
            """() => {
                const h = Math.floor(window.innerHeight * 0.3);
                window.scrollTo(0, h);
                setTimeout(() => window.scrollTo(0, 0), 300);
            }"""
        )
        await asyncio.sleep(0.5)
        # Add random delay to avoid bot detection
        await self._add_delay()
        return page

    async def _add_delay(self) -> None:
        """Add a random delay to simulate human behavior.

        Uses a bursty distribution: 60% quick (0.3–0.8s), 30% normal (1.0–2.5s),
        10% slow/hesitation (3.0–5.0s).
        """
        r = random.random()
        if r < 0.6:
            delay = random.uniform(0.3, 0.8)
        elif r < 0.9:
            delay = random.uniform(1.0, 2.5)
        else:
            delay = random.uniform(3.0, 5.0)
        await asyncio.sleep(delay)

    async def _simulate_mouse_movement(self, page: Page) -> None:
        """Move the real cursor along a short, jittered path (trusted input).

        Uses the CDP input pipeline (``page.mouse.move``) so events carry
        ``isTrusted === true``. Synthetic ``dispatchEvent`` mouse events are
        ``isTrusted === false`` and are explicitly flagged by behavioural
        detectors (e.g. brotector's ``Input.untrusted``), so we never use them.
        """
        try:
            target_x = random.randint(80, 1000)
            target_y = random.randint(80, 600)
            await page.mouse.move(target_x, target_y, steps=random.randint(5, 12))
        except Exception:
            pass

    async def screenshot(
        self,
        page: Page,
        full_page: bool = False,
        session_id: str | None = None,
    ) -> tuple[bytes, str]:
        """Take a screenshot of the page and save it to disk.

        Args:
            page: The page to screenshot.
            full_page: Whether to capture the full page.
            session_id: The session ID. Used for default path generation.

        Returns:
            Tuple of (raw PNG bytes, file path where the screenshot was saved).
        """
        # Capture as bytes
        screenshot_bytes = await page.screenshot(full_page=full_page)

        # Save to disk
        timestamp = asyncio.get_event_loop().time()
        session_prefix = session_id[:8] if session_id else "default"
        path = os.path.join(
            self._screenshot_dir,
            f"screenshot_{session_prefix}_{timestamp:.0f}.png",
        )
        os.makedirs(self._screenshot_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(screenshot_bytes)
        logger.info("Screenshot saved to %s", path)

        return screenshot_bytes, path

    async def get_interactables(self, page: Page) -> list[dict[str, Any]]:
        """Extract all interactable elements from the page.

        Returns a structured list of clickable and fillable elements with
        selectors, labels, text, type, and visibility state so the LLM
        can pick the right target without guessing.

        Args:
            page: The page to scan.

        Returns:
            List of dicts with keys: index, tag, type, text, name, id,
            placeholder, selector, visible.
        """
        elements = await page.evaluate(
            """() => {
                const results = [];
                let idx = 0;

                // Build a unique CSS selector for an element
                function buildSelector(el) {
                    if (el.id) return '#' + CSS.escape(el.id);
                    const parts = [];
                    let node = el;
                    while (node && node.nodeType === Node.ELEMENT_NODE) {
                        let sel = node.tagName.toLowerCase();
                        if (node.className && typeof node.className === 'string') {
                            const cls = node.className.trim().split(/\\s+/).slice(0, 2).map(c => '.' + CSS.escape(c)).join('');
                            sel += cls;
                        }
                        parts.unshift(sel);
                        if (parts.join(' > ').length > 80) break;
                        node = node.parentElement;
                    }
                    return parts.join(' > ');
                }

                function isVisible(el) {
                    if (!el) return false;
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    return rect.width > 0 && rect.height > 0
                        && style.display !== 'none'
                        && style.visibility !== 'hidden'
                        && style.opacity !== '0'
                        && !el.hasAttribute('disabled');
                }

                // Collect clickable elements
                const clickSelectors = [
                    'a[href]', 'button', 'input[type="submit"]',
                    'input[type="button"]', 'input[type="reset"]',
                    '[role="button"]', '[role="link"]',
                    'summary', 'details summary',
                ];
                for (const sel of clickSelectors) {
                    for (const el of document.querySelectorAll(sel)) {
                        if (!el.closest('details') || el.tagName === 'SUMMARY') {
                            results.push({
                                index: idx++,
                                tag: el.tagName.toLowerCase(),
                                type: 'clickable',
                                text: (el.textContent || '').trim().substring(0, 120),
                                name: el.getAttribute('name') || '',
                                id: el.id || '',
                                placeholder: el.getAttribute('placeholder') || '',
                                href: el.getAttribute('href') || '',
                                selector: buildSelector(el),
                                visible: isVisible(el),
                            });
                        }
                    }
                }

                // Collect fillable elements
                const fillSelectors = [
                    'input[type="text"]', 'input[type="search"]',
                    'input[type="email"]', 'input[type="password"]',
                    'input[type="number"]', 'input[type="tel"]',
                    'input[type="url"]', 'input:not([type])',
                    'textarea', '[contenteditable="true"]',
                    'select',
                ];
                for (const sel of fillSelectors) {
                    for (const el of document.querySelectorAll(sel)) {
                        results.push({
                            index: idx++,
                            tag: el.tagName.toLowerCase(),
                            type: 'fillable',
                            text: (el.textContent || '').trim().substring(0, 120),
                            name: el.getAttribute('name') || '',
                            id: el.id || '',
                            placeholder: el.getAttribute('placeholder') || '',
                            value: el.value || '',
                            selector: buildSelector(el),
                            visible: isVisible(el),
                        });
                    }
                }

                return results;
            }"""
        )
        logger.info("Extracted %d interactable elements", len(elements))
        return elements

    async def get_content(self, page: Page) -> str:
        """Get the page content as markdown-formatted text.

        Fetches the HTML content and converts it to clean markdown using html2text.
        This provides a more readable, LLM-friendly representation of page content
        compared to raw HTML.

        Args:
            page: The page to get content from.

        Returns:
            The page content as markdown-formatted text.
        """
        html_content = await page.content()
        
        # Convert HTML to markdown using html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0  # Don't wrap text
        h.unicode_snob = True
        h.inline_links = True
        h.protect_links = True
        
        markdown_content = h.handle(html_content)
        
        logger.info("Page content converted to markdown (%d bytes HTML -> %d bytes markdown)",
                    len(html_content), len(markdown_content))
        return markdown_content

    async def get_accessibility_snapshot(
        self,
        page: Page,
        depth: int | None = None,
        max_length: int | None = None,
    ) -> str:
        """Get ARIA snapshot of the page for structured page understanding.

        Uses mode='ai' to include [ref=eN] element references for later targeting.
        Returns a YAML-like string representing the accessibility tree.

        Args:
            page: The page to snapshot.
            depth: Maximum tree depth. None for full depth.
            max_length: If set, truncate output to this many characters.

        Returns:
            The ARIA snapshot string, optionally truncated.
        """
        snapshot = await page.aria_snapshot(
            depth=depth,
            mode="ai",
        )
        logger.info("ARIA snapshot captured (%d chars)", len(snapshot))
        if max_length and len(snapshot) > max_length:
            snapshot = snapshot[:max_length] + "\n... [truncated]"
        return snapshot

    async def get_page_state(
        self,
        page: Page,
        max_length: int | None = None,
    ) -> dict:
        """Get unified page state for rich LLM context.

        Returns a dict with: url, title, accessibility snapshot, interactables count,
        and heading hierarchy.

        Args:
            page: The page to inspect.
            max_length: If set, truncate accessibility snapshot to this many characters.

        Returns:
            Dict with url, title, accessibility (str), interactables_count (int),
            scroll_position (dict), and headings (list of dict).
        """
        interactables = await self.get_interactables(page)
        accessibility = await self.get_accessibility_snapshot(page, max_length=max_length)

        scroll_info = await page.evaluate(
            "() => ({top: window.scrollY, left: window.scrollX, height: document.body.scrollHeight})"
        )

        headings = self._extract_headings_from_aria_snapshot(accessibility)

        return {
            "url": page.url,
            "title": await page.title(),
            "accessibility": accessibility,
            "interactables_count": len(interactables),
            "scroll_position": scroll_info,
            "headings": headings,
        }

    def _extract_headings_from_aria_snapshot(
        self,
        snapshot: str,
    ) -> list[dict]:
        """Extract headings from an ARIA snapshot string.

        Parses lines matching: heading "Title" [level=N] [ref=eXX]

        Args:
            snapshot: The ARIA snapshot string.

        Returns:
            List of dicts with 'level' (int), 'text' (str), and 'ref' (str or None).
        """
        headings = []
        pattern = re.compile(r'heading\s+"([^"]*)"\s+\[level=(\d+)\](?:\s+\[ref=(e\d+)\])?')
        for match in pattern.finditer(snapshot):
            headings.append({
                "text": match.group(1),
                "level": int(match.group(2)),
                "ref": match.group(3),
            })
        return headings

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
        """Click an element with scroll-into-view and mouse-movement simulation.

        Tries visible elements first, then falls back to force-click if needed.

        Args:
            page: The page containing the element.
            selector: The CSS selector for the element.
        """
        # Scroll a VISIBLE matching element into view
        scrolled = await page.evaluate(
            """(sel) => {
                const els = document.querySelectorAll(sel);
                for (const el of els) {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    if (rect.width > 0 && rect.height > 0
                        && style.display !== 'none'
                        && style.visibility !== 'hidden'
                        && style.opacity !== '0') {
                        el.scrollIntoView({block: 'center', behavior: 'smooth'});
                        return true;
                    }
                }
                // Fallback: scroll the first match even if hidden
                if (els.length > 0) {
                    els[0].scrollIntoView({block: 'center', behavior: 'smooth'});
                }
                return false;
            }""",
            selector,
        )

        # Simulate mouse movement before clicking
        await self._simulate_mouse_movement(page)

        # Try clicking — prefer visible elements
        try:
            locator = page.locator(selector)
            visible_count = await locator.filter(visible=True).count()
            if visible_count > 0:
                await locator.filter(visible=True).first.click(timeout=5000)
            else:
                await locator.first.click(timeout=5000)
        except Exception:
            # Last resort: force-click the first match
            await page.locator(selector).first.click(timeout=5000, force=True)

        logger.info("Clicked element: %s (visible=%s)", selector, scrolled)

    async def fill(self, page: Page, selector: str, value: str) -> None:
        """Fill an input field with mouse-movement simulation.

        Tries visible elements first, then falls back to force-fill if needed.

        Args:
            page: The page containing the input.
            selector: The CSS selector for the input.
            value: The value to fill.
        """
        # Scroll a VISIBLE matching element into view
        await page.evaluate(
            """(sel) => {
                const els = document.querySelectorAll(sel);
                for (const el of els) {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    if (rect.width > 0 && rect.height > 0
                        && style.display !== 'none'
                        && style.visibility !== 'hidden'
                        && style.opacity !== '0') {
                        el.scrollIntoView({block: 'center', behavior: 'smooth'});
                        return true;
                    }
                }
                if (els.length > 0) {
                    els[0].scrollIntoView({block: 'center', behavior: 'smooth'});
                }
                return false;
            }""",
            selector,
        )

        # Simulate mouse movement before filling
        await self._simulate_mouse_movement(page)

        # Try filling — prefer visible elements
        try:
            locator = page.locator(selector)
            visible_count = await locator.filter(visible=True).count()
            if visible_count > 0:
                await locator.filter(visible=True).first.fill(value, timeout=5000)
            else:
                await locator.first.fill(value, timeout=5000)
        except Exception:
            # Last resort: force-fill the first match
            await page.locator(selector).first.fill(value, timeout=5000, force=True)

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
        """Check if the browser pool is running.

        Returns:
            True if the browser pool is initialized, False otherwise.
        """
        return len(self._browser_pool) > 0

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


# Lazy singleton — created on first access to avoid side effects at import time
_browser_manager: BrowserManager | None = None


def get_browser_manager() -> BrowserManager:
    """Get (or create) the global browser manager instance."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
    return _browser_manager


# Backwards compat alias — deprecated, use get_browser_manager()
browser_manager = get_browser_manager()
