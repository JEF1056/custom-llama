"""Tests for file management UI endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path


@pytest.fixture
def tmp_file_dir(tmp_path):
    """Create a temporary directory and set it as FILE_OUTPUT_DIR."""
    import src.config as cfg
    orig = cfg.settings.FILE_OUTPUT_DIR
    cfg.settings.FILE_OUTPUT_DIR = str(tmp_path)
    cfg.settings.FILE_BASE_URL = "http://test/files"
    yield tmp_path
    cfg.settings.FILE_OUTPUT_DIR = orig


@pytest.fixture
def file_app(tmp_file_dir):
    """Create a minimal Starlette app with the file management routes."""
    from src.server import (
        serve_file_ui, serve_file, list_files_api,
        upload_file, delete_file_api,
    )

    app = Starlette(
        routes=[
            Route("/files", endpoint=serve_file_ui, methods=["GET"]),
            Route("/files/{filename}", endpoint=serve_file, methods=["GET"]),
            Route("/api/files", endpoint=list_files_api, methods=["GET"]),
            Route("/api/files/upload", endpoint=upload_file, methods=["POST"]),
            Route("/api/files/{filename}", endpoint=delete_file_api, methods=["DELETE"]),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ],
    )
    return app


@pytest.mark.asyncio
async def test_file_ui_serves_html(file_app):
    """GET /files returns HTML page."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.get("/files")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "File Manager" in resp.text
    assert "tailwindcss" in resp.text


@pytest.mark.asyncio
async def test_list_files_empty(file_app, tmp_file_dir):
    """GET /api/files returns empty list when no files exist."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/files")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_files_with_files(file_app, tmp_file_dir):
    """GET /api/files returns list of files with metadata."""
    # Create a test file
    test_file = tmp_file_dir / "test.txt"
    test_file.write_text("hello")

    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/files")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "test.txt"
    assert data[0]["size_bytes"] == 5
    assert "B" in data[0]["size"]  # human-readable size
    assert "2025" in data[0]["modified"] or "2026" in data[0]["modified"]


@pytest.mark.asyncio
async def test_upload_file(file_app, tmp_file_dir):
    """POST /api/files/upload uploads a file successfully."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/files/upload",
            files={"file": ("uploaded.txt", b"test content", "text/plain")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["filename"] == "uploaded.txt"
    assert data["size"] == 12
    assert (tmp_file_dir / "uploaded.txt").read_bytes() == b"test content"


@pytest.mark.asyncio
async def test_upload_no_file_field(file_app):
    """POST /api/files/upload without file field -> 400."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.post("/api/files/upload", data={})
    assert resp.status_code == 400
    assert "No 'file' field" in resp.json()["error"]


@pytest.mark.asyncio
async def test_upload_invalid_filename(file_app):
    """POST /api/files/upload with path traversal -> 400."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        # Path traversal attempt
        resp = await client.post(
            "/api/files/upload",
            files={"file": ("../evil.txt", b"bad", "text/plain")},
        )
    assert resp.status_code == 400
    assert "Invalid filename" in resp.json()["error"]


@pytest.mark.asyncio
async def test_upload_slash_in_filename(file_app):
    """POST /api/files/upload with slash in filename -> 400."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/files/upload",
            files={"file": ("sub/dir.txt", b"bad", "text/plain")},
        )
    assert resp.status_code == 400
    assert "Invalid filename" in resp.json()["error"]


@pytest.mark.asyncio
async def test_delete_file(file_app, tmp_file_dir):
    """DELETE /api/files/{filename} deletes a file successfully."""
    test_file = tmp_file_dir / "delete_me.txt"
    test_file.write_text("to be deleted")

    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.delete("/api/files/delete_me.txt")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert not (tmp_file_dir / "delete_me.txt").exists()


@pytest.mark.asyncio
async def test_delete_file_not_found(file_app):
    """DELETE /api/files/{filename} for non-existent file -> 404."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.delete("/api/files/nonexistent.txt")
    assert resp.status_code == 404
    assert "not found" in resp.json()["error"]


@pytest.mark.asyncio
async def test_delete_path_traversal_blocked(file_app):
    """DELETE /api/files/{filename} with path traversal -> blocked (404 or 400)."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        # Starlette normalizes URLs so ../etc/passwd becomes etc/passwd,
        # which doesn't match the route pattern (404). Either way, traversal is blocked.
        resp = await client.delete("/api/files/../etc/passwd")
    assert resp.status_code in (400, 404)


@pytest.mark.asyncio
async def test_file_download(file_app, tmp_file_dir):
    """GET /files/{filename} serves a file."""
    test_file = tmp_file_dir / "download.txt"
    test_file.write_text("downloadable content")

    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.get("/files/download.txt")
    assert resp.status_code == 200
    assert resp.text == "downloadable content"


@pytest.mark.asyncio
async def test_file_download_not_found(file_app):
    """GET /files/{filename} for non-existent file -> 404."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.get("/files/nonexistent.txt")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_then_list(file_app, tmp_file_dir):
    """Integration: upload a file, then verify it appears in list."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        # Upload
        resp = await client.post(
            "/api/files/upload",
            files={"file": ("integration_test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 200

        # List
        resp = await client.get("/api/files")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "integration_test.txt"
        assert data[0]["size_bytes"] == 11


@pytest.mark.asyncio
async def test_upload_then_delete(file_app, tmp_file_dir):
    """Integration: upload a file, then delete it."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        # Upload
        resp = await client.post(
            "/api/files/upload",
            files={"file": ("del_test.txt", b"delete me", "text/plain")},
        )
        assert resp.status_code == 200

        # Delete
        resp = await client.delete("/api/files/del_test.txt")
        assert resp.status_code == 200

        # Verify it's gone
        resp = await client.get("/api/files")
        assert resp.json() == []


@pytest.mark.asyncio
async def test_file_ui_before_download_route(file_app, tmp_file_dir):
    """GET /files (exact) returns HTML, not a 404 from /files/{filename} pattern."""
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.get("/files")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_format_size_bytes():
    """Test _format_size function with various sizes."""
    from src.server import _format_size
    assert _format_size(0) == "0 B"
    assert _format_size(1) == "1 B"
    assert "KB" in _format_size(1500)
    assert "MB" in _format_size(1_500_000)
    assert "GB" in _format_size(1_500_000_000)


@pytest.mark.asyncio
async def test_auth_skips_file_ui(file_app):
    """GET /files and /api/files don't require auth (no middleware in test app)."""
    # These endpoints are tested without auth middleware to verify they work
    # independently. The real server wraps them with ApiKeyAuthMiddleware,
    # but the middleware skips /files (exact), /api/files paths.
    async with AsyncClient(
        transport=ASGITransport(app=file_app), base_url="http://test"
    ) as client:
        resp = await client.get("/files")
        assert resp.status_code == 200
        resp = await client.get("/api/files")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_skips_new_routes():
    """Verify that new routes are not blocked by auth middleware."""
    import src.config as cfg
    from src.server import create_app
    from src.server import ApiKeyAuthMiddleware

    orig = cfg.settings.MCP_API_KEY
    cfg.settings.MCP_API_KEY = "secret-key"
    try:
        # Use a minimal Starlette app with just the new routes + auth middleware
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.middleware import Middleware
        from starlette.middleware.cors import CORSMiddleware

        from src.server import serve_file_ui, list_files_api

        app = Starlette(
            routes=[
                Route("/files", endpoint=serve_file_ui, methods=["GET"]),
                Route("/api/files", endpoint=list_files_api, methods=["GET"]),
            ],
            middleware=[
                Middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                ),
                Middleware(ApiKeyAuthMiddleware),
            ],
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # No auth header — should still work because middleware skips these paths
            resp = await client.get("/files")
            assert resp.status_code == 200
            resp = await client.get("/api/files")
            assert resp.status_code == 200
    finally:
        cfg.settings.MCP_API_KEY = orig
