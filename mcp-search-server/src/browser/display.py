"""Virtual display (Xvfb) lifecycle management for headful browser anti-detection.

Running Chrome *headful* under a virtual display is the strongest anti-bot
posture (a real windowed browser has genuine outer dimensions, GPU path, and a
normal event loop). But the X server is a separate process that can die at any
time — most often killed by the host OOM killer under memory pressure. When it
dies, the X socket file goes *stale* (it remains on disk), so a naive
"does the socket exist?" check reports the display as available when it is not,
and every headed Chrome launch then fails with ``Missing X server or $DISPLAY``.

This module owns the display so the server can recover at runtime instead of
breaking: it does a *true* liveness probe (an X11 protocol handshake, not a
socket-file check), (re)starts Xvfb on demand, and tracks the child it spawns so
it can be reaped cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import socket
import subprocess
import time

logger = logging.getLogger(__name__)

def x_display_alive(display: str, timeout: float = 2.0) -> bool:
    """Return True only if a live X server is *listening* on ``display``.

    A running Xvfb keeps a unix socket open at ``/tmp/.X11-unix/Xn`` and accepts
    connections. When Xvfb dies the socket stops accepting (a SIGKILL can leave
    the socket *file* behind, but with no listener behind it), so a successful
    ``connect()`` is a dependency-free proof that a live X server is serving the
    display. We deliberately do NOT rely on reading a protocol handshake: the
    server does not transmit on connect, so a read would block.

    Args:
        display: X display name, e.g. ``":99"``.
        timeout: Seconds to wait for the connection.

    Returns:
        True if a live X server accepted the connection, else False.
    """
    if not display:
        return False
    num = display.lstrip(":")
    if not num:
        return False
    sock_path = f"/tmp/.X11-unix/X{num}"
    if not os.path.exists(sock_path):
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(sock_path)
        return True
    except (OSError, socket.timeout):
        return False
    finally:
        sock.close()


def in_docker() -> bool:
    """Best-effort detection of running inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="ascii") as fh:
            return "docker" in fh.read().lower()
    except OSError:
        return False


class DisplayManager:
    """Starts, probes, and recovers a virtual X display (Xvfb).

    The manager is intentionally conservative: it never starts a second Xvfb
    while a live one already serves the display (e.g. one started by the
    container entrypoint), and it gives up on headed mode after repeated
    failures so the caller can fall back to headless rather than hang.
    """

    def __init__(self, display: str | None = None, max_starts: int = 3):
        self.display = display or os.environ.get("DISPLAY", ":99")
        self.max_starts = max_starts
        self._xvfb: subprocess.Popen | None = None
        self._lock = asyncio.Lock()
        self._start_failures = 0

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if a live X server is currently answering on the display."""
        return x_display_alive(self.display)

    @property
    def managed_pid(self) -> int | None:
        """PID of the Xvfb we spawned, if any (else None)."""
        if self._xvfb is not None and self._xvfb.poll() is None:
            return self._xvfb.pid
        return None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def ensure(self) -> bool:
        """Ensure a live display, starting Xvfb if necessary.

        Returns:
            True if a live X server is available afterwards, else False (the
            caller should fall back to headless).
        """
        async with self._lock:
            if x_display_alive(self.display):
                self._start_failures = 0
                return True
            return await self._start_xvfb()

    def stop(self) -> None:
        """Terminate the Xvfb we spawned (if any). Idempotent and safe to call
        from a sync context (e.g. shutdown)."""
        if self._xvfb is not None and self._xvfb.poll() is None:
            with contextlib.suppress(Exception):
                self._xvfb.terminate()
                self._xvfb.wait(timeout=5)
        self._xvfb = None

    # ── Internals ────────────────────────────────────────────────────────────

    async def _start_xvfb(self) -> bool:
        if self._start_failures >= self.max_starts:
            logger.error(
                "Xvfb has failed to start %d times; giving up on headed mode",
                self._start_failures,
            )
            return False
        if shutil.which("Xvfb") is None:
            logger.error("Xvfb binary not found on PATH; cannot run headed")
            self._start_failures = self.max_starts
            return False

        self._reap_stale()
        os.makedirs("/tmp/.X11-unix", exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod("/tmp/.X11-unix", 0o1777)

        cmd = [
            "Xvfb", self.display,
            "-screen", "0", "1920x1080x24",
            "-ac", "+extension", "GLX", "+render",
            "-noreset", "-nolisten", "tcp",
        ]
        try:
            self._xvfb = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as exc:  # noqa: BLE001 - report and fall back to headless
            logger.error("Failed to spawn Xvfb: %s", exc)
            self._xvfb = None
            self._start_failures += 1
            return False

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if x_display_alive(self.display):
                self._start_failures = 0
                logger.info(
                    "Xvfb (re)started on %s (pid=%s)", self.display, self._xvfb.pid
                )
                return True
            if self._xvfb.poll() is not None:
                logger.warning(
                    "Xvfb exited early (code=%s) while starting on %s",
                    self._xvfb.returncode, self.display,
                )
                break
            await asyncio.sleep(0.1)

        self._start_failures += 1
        logger.warning(
            "Xvfb not ready on %s after attempt %d/%d",
            self.display, self._start_failures, self.max_starts,
        )
        return False

    def _reap_stale(self) -> None:
        """Kill a dead Xvfb we own and clear its stale state before restarting.

        A dead Xvfb leaves two stale artifacts that block a fresh start: the X
        socket (``/tmp/.X11-unix/Xn``) and, crucially, the lock file
        (``/tmp/.Xn-lock``). Xvfb refuses to start while the lock exists
        ("Server is already active for display n"), so both must be removed.
        """
        if self._xvfb is not None:
            if self._xvfb.poll() is None:
                # We shouldn't normally reach here with a live one we own, but be safe.
                with contextlib.suppress(Exception):
                    self._xvfb.terminate()
                    self._xvfb.wait(timeout=5)
            self._xvfb = None
        num = self.display.lstrip(":")
        for stale in (f"/tmp/.X11-unix/X{num}", f"/tmp/.X{num}-lock"):
            with contextlib.suppress(OSError):
                os.unlink(stale)
