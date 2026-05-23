import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI, Response
from src.server import create_app, ApiKeyAuthMiddleware


class MockMCPApp:
    """Mock app that returns 200 for /mcp and /sse endpoints."""
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope["path"]
        if path == "/mcp" and scope["method"] == "POST":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"ok": true}',
            })
        elif path == "/sse" and scope["method"] == "GET":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/event-stream"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'data: ok\n\n',
            })
        else:
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'not found',
            })


def make_test_app():
    """Create a test app with mock endpoints wrapped by auth middleware."""
    app = FastAPI()
    app.add_middleware(ApiKeyAuthMiddleware)
    app.mount("/", MockMCPApp())
    return app


# --- Auth rejection tests ---

@pytest.mark.asyncio
async def test_auth_rejects_missing_key():
    """POST /mcp without Authorization header -> 401."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        app = make_test_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/mcp")
        assert resp.status_code == 401
    finally:
        cfg.settings.MCP_API_KEY = orig


@pytest.mark.asyncio
async def test_auth_rejects_wrong_key():
    """POST /mcp with wrong Bearer token -> 401."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        app = make_test_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/mcp",
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert resp.status_code == 401
    finally:
        cfg.settings.MCP_API_KEY = orig


# --- Auth bypass tests ---

@pytest.mark.asyncio
async def test_auth_skips_health():
    """GET /health -> 200 even with auth enabled (no key needed)."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        app = make_test_app()
        # /health doesn't exist in our mock, but the middleware should
        # skip auth before the request reaches the mock app.
        # We verify the middleware doesn't block /health by checking
        # that the mock app receives the request (returns 404, not 401).
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        # 404 means the middleware let it through to the mock app
        # (which doesn't have /health). If auth blocked it, we'd get 401.
        assert resp.status_code == 404
    finally:
        cfg.settings.MCP_API_KEY = orig


@pytest.mark.asyncio
async def test_auth_skips_sse_get():
    """GET /sse -> 200 even with auth enabled (no key needed)."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        app = make_test_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/sse")
        assert resp.status_code == 200
    finally:
        cfg.settings.MCP_API_KEY = orig


@pytest.mark.asyncio
async def test_auth_allows_correct_key_mcp():
    """POST /mcp with correct Bearer token -> 200."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        app = make_test_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/mcp",
                headers={"Authorization": "Bearer secret-key"},
            )
        assert resp.status_code == 200
    finally:
        cfg.settings.MCP_API_KEY = orig


@pytest.mark.asyncio
async def test_auth_allows_correct_key_files():
    """GET /files/test.txt with correct Bearer token -> 200."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        # For /files/* the mock returns 404, but the middleware should
        # let the request through (not 401).
        app = make_test_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/files/test.txt",
                headers={"Authorization": "Bearer secret-key"},
            )
        # 404 means middleware passed it through; 401 would mean blocked
        assert resp.status_code == 404
    finally:
        cfg.settings.MCP_API_KEY = orig


@pytest.mark.asyncio
async def test_auth_rejects_files_without_key():
    """GET /files/test.txt without key -> 401."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        app = make_test_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/files/test.txt")
        assert resp.status_code == 401
    finally:
        cfg.settings.MCP_API_KEY = orig


@pytest.mark.asyncio
async def test_auth_disabled_when_empty():
    """When MCP_API_KEY is empty, auth is disabled entirely."""
    import src.config as cfg
    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = ""
    try:
        app = make_test_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # /mcp without key should succeed (not 401)
            resp = await client.post("/mcp")
        # Mock returns 200 for /mcp POST
        assert resp.status_code == 200
    finally:
        cfg.settings.MCP_API_KEY = orig

