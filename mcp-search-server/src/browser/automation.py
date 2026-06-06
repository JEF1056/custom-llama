"""Browser automation using Playwright with session management."""

import asyncio
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth

from src.config import settings

logger = logging.getLogger(__name__)

# Common timezones for fingerprint randomization
TIMEZONES = [
    "America/New_York",
    "America/Los_Angeles",
    "America/Chicago",
    "America/Denver",
    "Europe/London",
    "Europe/Berlin",
    "Europe/Paris",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Australia/Sydney",
]

# Common locales for fingerprint randomization
LOCALES = ["en-US", "en-GB", "en-CA", "de-DE", "fr-FR", "ja-JP", "zh-CN", "es-ES"]

# Common user agents for rotation
USER_AGENTS = [
    # Chrome — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    # Chrome — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    # Chrome — Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    # Edge — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
    # Edge — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Firefox — macOS (Gecko)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Firefox — Windows (Gecko)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    # Firefox — Linux (Gecko)
    "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
]

# JavaScript hardening script injected via add_init_script on every context.
# Covers: navigator hardening, permissions spoofing, chrome.runtime shell,
# __proto__ removal, getBoundingClientRect noise, and canvas/WebGL spoofing.
HARDENING_SCRIPT = """
Object.freeze;

// ── Navigator hardening ──────────────────────────────────────────────
// Remove webdriver flag that headless browsers expose
delete Object.getPrototypeOf(navigator).webdriver;

// PhantomJS / Nightmare fingerprints
if (window.callPhantom) { delete window.callPhantom; }
if (window._nightmare) { delete window._nightmare; }

// ── Permissions hardening ────────────────────────────────────────────
// Spoof Permissions API to always return granted for notifications
const _origQuery = navigator.permissions && navigator.permissions.query;
if (navigator.permissions) {
    navigator.permissions.query = function (descriptor) {
        if (descriptor && descriptor.name === "notifications") {
            return Promise.resolve({
                name: "notifications",
                state: "granted",
                onchange: null,
                addEventListener: Function.prototype,
                removeEventListener: Function.prototype,
                dispatchEvent: Function.prototype,
            });
        }
        return _origQuery ? _origQuery.call(navigator.permissions, descriptor) : Promise.reject(new Error("Not supported"));
    };
}

// ── Chrome runtime spoofing ──────────────────────────────────────────
// Inject a realistic chrome.runtime shell (non-functional stub)
if (!window.chrome) {
    window.chrome = {};
}
window.chrome.runtime = {
    id: "aapnipglbcbfoncpnmpnpenhpkbpkfgi",
    connect: function () { return { onMessage: { addListener: Function.prototype }, send: Function.prototype, disconnect: Function.prototype }; },
    sendMessage: function (/* message, callback */) { /* no-op */ },
    onMessage: {
        addListener: Function.prototype,
        removeListener: Function.prototype,
        hasListener: Function.prototype,
        hasListeners: function () { return false; },
    },
    getURL: function (path) { return "chrome-extension://" + window.chrome.runtime.id + "/" + path; },
    getManifest: function () {
        return {
            name: "Chrome Extension",
            version: "1.0",
            manifest_version: 3,
            permissions: ["storage"],
        };
    },
    getBackgroundPage: function (cb) { /* no-op */ },
    getPlatformInfo: function (cb) { if (cb) cb({ os: "win" }); },
};

// ── __proto__ removal ────────────────────────────────────────────────
// Some detectors check for __proto__ on navigator
delete navigator.__proto__;

// ── getBoundingClientRect noise ──────────────────────────────────────
// Add ±0.5px random noise to coordinates
coreFunc = Element.prototype.getBoundingClientRect;
Element.prototype.getBoundingClientRect = function () {
    const original = coreFunc.call(this);
    const noise = () => (Math.random() - 0.5) * 1.0;
    return {
        x: original.x + noise(),
        y: original.y + noise(),
        width: original.width,
        height: original.height,
        top: original.top + noise(),
        right: original.right + noise(),
        bottom: original.bottom + noise(),
        left: original.left + noise(),
    };
};

// ── Canvas spoofing ──────────────────────────────────────────────────
// Add 1–5 pixels of noise to getImageData output
const _origGetImageData = CanvasRenderingContext2D.prototype.getImageData;
CanvasRenderingContext2D.prototype.getImageData = function (sx, sy, sw, sh) {
    const imageData = _origGetImageData.call(this, sx, sy, sw, sh);
    const data = imageData.data;
    const noiseLevel = Math.floor(Math.random() * 5) + 1; // 1-5
    for (let i = 0; i < data.length; i += 4) {
        if (Math.random() > 0.9) {
            const noise = (Math.random() - 0.5) * noiseLevel;
            data[i] = Math.min(255, Math.max(0, data[i] + noise));
            data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
            data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
        }
    }
    return imageData;
};

// ── WebGL spoofing ───────────────────────────────────────────────────
// Add 1–5 pixels of noise to readPixels output
const _origReadPixels = WebGLRenderingContext.prototype.readPixels;
WebGLRenderingContext.prototype.readPixels = function (x, y, w, h, format, type, pixels, offset) {
    _origReadPixels.call(this, x, y, w, h, format, type, pixels, offset);
    if (pixels && pixels.length) {
        const noiseLevel = Math.floor(Math.random() * 5) + 1;
        for (let i = 0; i < pixels.length; i++) {
            if (Math.random() > 0.9) {
                const noise = (Math.random() - 0.5) * noiseLevel;
                pixels[i] = Math.min(255, Math.max(0, pixels[i] + noise));
            }
        }
    }
};
"""


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
    """Manages Playwright browser instances with anti-detection features and session support."""

    def __init__(self, screenshot_dir: str | None = None):
        """Initialize the browser manager.

        Args:
            screenshot_dir: Directory to save screenshots. Defaults to /app/mcp-files/screenshots.
        """
        self._playwright = None
        self._browser: Browser | None = None
        self._default_context: BrowserContext | None = None
        self._sessions: dict[str, BrowserSession] = {}
        self._screenshot_dir = screenshot_dir or settings.SCREENSHOT_DIR
        self._cleanup_task: asyncio.Task | None = None
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
        self._default_context = await self._create_hardened_context()
        await self._default_context.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        })
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Browser started")

    async def _create_hardened_context(self) -> BrowserContext:
        """Create a new browser context with randomized fingerprint properties.

        Randomizes viewport dimensions, timezone, and locale to avoid
        fingerprint-based detection.

        Returns:
            A new BrowserContext with randomized properties and stealth applied.
        """
        viewport = {
            "width": random.randint(120, 192) * 10,
            "height": random.randint(70, 108) * 10,
        }
        timezone = random.choice(TIMEZONES)
        locale = random.choice(LOCALES)

        context = await self._browser.new_context(
            viewport=viewport,
            user_agent=random.choice(USER_AGENTS),
            ignore_https_errors=True,
            timezone_id=timezone,
            locale=locale,
        )
        await Stealth().apply_stealth_async(context)
        await context.add_init_script(HARDENING_SCRIPT)
        logger.info(
            "Hardened context: viewport=%sx%s, tz=%s, locale=%s",
            viewport["width"],
            viewport["height"],
            timezone,
            locale,
        )
        return context

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
        # Cancel background cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("Browser stopped")

    async def create_session(self) -> str:
        """Create a new browser session.

        Returns:
            The session ID.
        """
        if not self.is_running:
            await self.start()

        session = BrowserSession()
        session.context = await self._create_hardened_context()
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
            session_id: The session ID. If None, returns the default context.

        Returns:
            The browser context, or None if not found.
        """
        if session_id:
            session = self._sessions.get(session_id)
            if session and session.is_active:
                self._touch_session(session_id)
                return session.context
            return None
        return self._default_context

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
        """Simulate a random mouse-movement event near the viewport center."""
        await page.evaluate(
            """() => {
                const x = Math.floor(Math.random() * 200) + 100;
                const y = Math.floor(Math.random() * 200) + 100;
                const el = document.elementFromPoint(x, y);
                if (el) {
                    el.dispatchEvent(new MouseEvent('mousemove', {
                        clientX: x, clientY: y,
                        bubbles: true, cancelable: true,
                    }));
                }
            }"""
        )

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
