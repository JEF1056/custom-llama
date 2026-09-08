"""Tests for the browser anti-detection / recovery helpers.

Covers the two runtime-recovery behaviours that keep the server working:
  * ``rewrite_local_url`` — lets a containerised browser reach dev servers on
    the host (the "can't access localhost URLs" fix),
  * the Xvfb display probe (``x_display_alive`` / ``DisplayManager``) that
    detects a dead "X server" so headed mode can recover (the crash fix).

These are pure-logic tests: no real browser or X server is required.
"""

import pytest

from src.browser.automation import rewrite_local_url
from src.browser.display import DisplayManager, in_docker, x_display_alive
from src.config import settings


# ── rewrite_local_url ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://localhost:3000/app", "http://host.docker.internal:3000/app"),
        ("http://127.0.0.1:8080/x?y=1#z", "http://host.docker.internal:8080/x?y=1#z"),
        ("localhost:5173", "http://host.docker.internal:5173"),
        ("http://[::1]:3000/", "http://host.docker.internal:3000/"),
        ("http://0.0.0.0:4000/", "http://host.docker.internal:4000/"),
        # Already the target host — left alone.
        ("http://host.docker.internal:3000/", "http://host.docker.internal:3000/"),
        # Non-local hosts are never touched.
        ("https://example.com/keep", "https://example.com/keep"),
        ("http://myapp.local:3000/", "http://myapp.local:3000/"),
    ],
)
def test_rewrite_local_url_in_docker(monkeypatch, url, expected):
    monkeypatch.setattr("src.browser.automation.in_docker", lambda: True)
    monkeypatch.setattr(settings, "BROWSER_REWRITE_LOCALHOST", True)
    monkeypatch.setattr(settings, "BROWSER_HOST_TARGET", "host.docker.internal")
    assert rewrite_local_url(url) == expected


def test_rewrite_local_url_disabled(monkeypatch):
    monkeypatch.setattr("src.browser.automation.in_docker", lambda: True)
    monkeypatch.setattr(settings, "BROWSER_REWRITE_LOCALHOST", False)
    assert rewrite_local_url("http://localhost:3000/") == "http://localhost:3000/"


def test_rewrite_local_url_outside_docker_unchanged(monkeypatch):
    # Outside Docker, localhost must keep working as-is.
    monkeypatch.setattr("src.browser.automation.in_docker", lambda: False)
    monkeypatch.setattr(settings, "BROWSER_REWRITE_LOCALHOST", True)
    assert rewrite_local_url("http://localhost:3000/") == "http://localhost:3000/"


def test_rewrite_local_url_empty():
    assert rewrite_local_url("") == ""


# ── X display liveness ───────────────────────────────────────────────────────

def test_x_display_alive_false_without_server():
    # Probe a display very unlikely to be in use; skip if one is actually live.
    import os
    display = ":5999"
    if os.path.exists(f"/tmp/.X11-unix/X{display.lstrip(':')}"):
        pytest.skip("an X server is running on the test display")
    assert x_display_alive(display) is False


def test_x_display_alive_rejects_blank():
    assert x_display_alive("") is False
    assert x_display_alive(":") is False


def test_in_docker_returns_bool():
    assert isinstance(in_docker(), bool)


def test_display_manager_reports_unavailable(monkeypatch):
    monkeypatch.setattr("src.browser.display.x_display_alive", lambda d, t=2.0: False)
    dm = DisplayManager(display=":99")
    assert dm.available is False
    assert dm.managed_pid is None
