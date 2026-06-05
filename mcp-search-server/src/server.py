"""MCP Server for web search and browser automation."""

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from src.config import settings
from src.tools.browser import browser_handler
from src.tools.calculator import calculator_handler
from src.tools.code_run import code_run_handler
from src.tools.deep_search import deep_search_handler
from src.tools.fetch import fetch_handler
from src.tools.file_ops import file_operations_handler
from src.tools.filetool import create_file_handler
from src.tools.http_request import http_request_handler
from src.tools.pptx_create import pptx_create_handler
from src.tools.pptx_edit import pptx_edit_handler
from src.tools.pptx_read import pptx_read_handler
from src.tools.pptx_slide_image import pptx_slide_image_handler
from src.tools.search import search_handler
from src.tools.time_now import time_now_handler
from src.tools.xlsx_create import xlsx_create_handler
from src.tools.xlsx_edit import xlsx_edit_handler
from src.tools.xlsx_read import xlsx_read_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware with per-IP rate limiting on failures.

    Reads MCP_API_KEY from settings.  When the key is empty, auth is skipped
    entirely (backward-compatible).  When set, requires
    `Authorization: Bearer <key>` on /mcp, /files/*, and /messages/ endpoints.
    The /health and GET /sse endpoints bypass auth.

    Rate limiting: 5 failures within 60 s per client IP → HTTP 429.
    Client IP is taken from the CF-Connecting-IP header (Cloudflare tunnel),
    falling back to request.client.host.
    """

    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX = 5  # failures before 429

    def __init__(self, app):
        super().__init__(app)
        self._failures: dict[str, list[float]] = {}

    def _get_client_ip(self, request: Request) -> str:
        return request.headers.get("CF-Connecting-IP", request.client.host)

    def _record_failure(self, ip: str) -> tuple[int, int]:
        """Record an auth failure and return (count, is_rate_limited)."""
        now = time.time()
        if ip not in self._failures:
            self._failures[ip] = []
        # Prune entries older than the window
        self._failures[ip] = [
            t for t in self._failures[ip] if now - t < self.RATE_LIMIT_WINDOW
        ]
        self._failures[ip].append(now)
        count = len(self._failures[ip])
        return count, count > self.RATE_LIMIT_MAX

    async def dispatch(self, request: Request, call_next):
        # Skip auth entirely when no key is configured
        if not settings.MCP_API_KEY:
            return await call_next(request)

        path = request.scope["path"]

        # Public endpoints — no auth
        if path == "/health":
            return await call_next(request)
        if path == "/sse" and request.method == "GET":
            return await call_next(request)

        # Protected endpoints
        if path == "/" or path.startswith("/files/") or path == "/messages/" or path.startswith("/api/files"):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    media_type="application/json",
                )
            token = auth_header[7:]
            if token != settings.MCP_API_KEY:
                ip = self._get_client_ip(request)
                count, rate_limited = self._record_failure(ip)
                if rate_limited:
                    return Response(
                        content=json.dumps({"error": "Too many requests"}),
                        status_code=429,
                        media_type="application/json",
                    )
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    media_type="application/json",
                )

        return await call_next(request)


def create_server() -> FastMCP:
    """Create and configure the MCP server.

    Returns:
        Configured MCP server instance.
    """
    server = FastMCP(
        name="mcp-search-server",
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        streamable_http_path="/",
        stateless_http=True,
        instructions=(
            "Web search, browser automation, HTTP, code execution, files, math, time, "
            "and Office document tools.\n\n"
            "Use when: agent needs live web data, URL fetching, API calls, Python computation, "
            "file I/O, math, timezone conversion, or spreadsheet/presentation generation.\n\n"
            "Prefer browser_* for JS-heavy pages; http_request for REST APIs.\n\n"
            "Documents (xlsx/pptx): build iteratively — create minimal → read to verify → "
            "edit incrementally. Use pptx_slide_image to visually verify slides.\n\n"
            "Common chains:\n"
            "- search/fetch → code_run → create_file/xlsx_create\n"
            "- http_request → code_run → xlsx_create → xlsx_read\n"
            "- pptx_create → pptx_edit → pptx_slide_image\n"
            "- calculator (symbolic) or code_run (numeric) for math"
        ),
    )
    return server


def register_tools(server: FastMCP) -> None:
    """Register all MCP tools with the server.

    Args:
        server: The MCP server instance.
    """
    search_handler(server)
    fetch_handler(server)
    deep_search_handler(server)
    browser_handler(server)
    create_file_handler(server)
    http_request_handler(server)
    code_run_handler(server)
    file_operations_handler(server)
    time_now_handler(server)
    calculator_handler(server)
    xlsx_create_handler(server)
    xlsx_read_handler(server)
    xlsx_edit_handler(server)
    pptx_create_handler(server)
    pptx_read_handler(server)
    pptx_edit_handler(server)
    pptx_slide_image_handler(server)


def register_resources(server: FastMCP) -> None:
    """Register MCP Resources and ResourceTemplates.

    Registers a resource template for reading files created by the create_file
    tool or screenshots saved by browser_screenshot / browser_monitor.
    The LLM can read these via the resources/read RPC call.
    """

    @server.resource("file:///app/mcp-files/{filename}")
    def read_file(filename: str) -> str:
        """Read a file from the output directory.

        Args:
            filename: Name of the file to read.
        """
        file_path = Path(settings.FILE_OUTPUT_DIR) / filename
        if not file_path.exists():
            raise ValueError(f"File not found: {filename}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {filename}")
        # Try text first, fall back to base64 for binary
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            import base64
            return base64.b64encode(file_path.read_bytes()).decode("utf-8")

    logger.info("Registered file resource template: file:///app/mcp-files/{filename}")


def healthcheck(request: Request) -> Response:
    """Simple healthcheck endpoint."""
    return Response(content="OK", media_type="text/plain", status_code=200)


def _format_size(size_bytes: int) -> str:
    """Format a file size in bytes as a human-readable string."""
    n = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            if unit == "B":
                return f"{int(size_bytes)} B"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _list_files_metadata() -> list[dict]:
    """List files in the output directory with metadata."""
    output_dir = Path(settings.FILE_OUTPUT_DIR)
    entries = []
    for entry in sorted(output_dir.iterdir(), key=lambda e: e.stat().st_mtime, reverse=True):
        if not entry.is_file():
            continue
        stat = entry.stat()
        entries.append({
            "name": entry.name,
            "size": _format_size(stat.st_size),
            "size_bytes": stat.st_size,
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
            "type": "application/octet-stream",
        })
    return entries


async def serve_file(request: Request) -> Response:
    """Serve a file from the output directory.

    Exposes files created by MCP tools (create_file, xlsx_create, pptx_create, etc.)
    as downloadable resources via the FILE_BASE_URL.
    """
    filename = request.path_params["filename"]
    file_path = Path(settings.FILE_OUTPUT_DIR) / filename
    if not file_path.exists() or not file_path.is_file():
        return Response(content="Not found", status_code=404)
    return FileResponse(str(file_path))


async def list_files_api(request: Request) -> Response:
    """List files in the output directory as JSON."""
    return JSONResponse(_list_files_metadata())


async def upload_file(request: Request) -> Response:
    """Upload a file via multipart form data."""
    form = await request.form()
    file = form.get("file")
    if file is None:
        return JSONResponse({"error": "No 'file' field in form data"}, status_code=400)

    filename = file.filename
    if not filename or "/" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)

    # 50 MB limit
    content = file.file.read()
    file.file.seek(0)
    if len(content) > 50 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 50 MB)"}, status_code=400)

    file_path = Path(settings.FILE_OUTPUT_DIR) / filename
    file_path.write_bytes(content)

    return JSONResponse({
        "status": "success",
        "filename": filename,
        "size": file_path.stat().st_size,
        "download_url": f"{settings.FILE_BASE_URL}/files/{filename}",
    })


async def delete_file_api(request: Request) -> Response:
    """Delete a file from the output directory."""
    filename = request.path_params["filename"]
    if "/" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)

    file_path = Path(settings.FILE_OUTPUT_DIR) / filename
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    file_path.unlink()
    return JSONResponse({"status": "success", "message": f"Deleted file: {filename}"})


FILE_UI_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>File Manager</title>
<script src="https://cdn.tailwindcss.com"></script>
</head><body class="bg-gray-50 min-h-screen">
<div class="max-w-4xl mx-auto px-4 py-8">
<h1 class="text-2xl font-bold text-gray-800 mb-6">File Manager</h1>

<div id="dropzone" class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-6
     hover:border-blue-400 transition-colors cursor-pointer"
     ondragover="event.preventDefault();this.classList.add('border-blue-400','bg-blue-50')"
     ondragleave="this.classList.remove('border-blue-400','bg-blue-50')"
     ondrop="handleDrop(event)" onclick="document.getElementById('fileinput').click()">
  <p class="text-gray-500">Drag &amp; drop files here or <span class="text-blue-500 underline">click to browse</span></p>
  <input type="file" id="fileinput" class="hidden" onchange="handleFileSelect(event)">
</div>

<div id="status" class="hidden mb-4 p-3 rounded-lg text-sm"></div>

<table class="w-full bg-white rounded-lg shadow overflow-hidden">
<thead class="bg-gray-100"><tr>
<th class="text-left px-4 py-2 text-sm font-semibold text-gray-600">Name</th>
<th class="text-left px-4 py-2 text-sm font-semibold text-gray-600">Size</th>
<th class="text-left px-4 py-2 text-sm font-semibold text-gray-600">Modified</th>
<th class="text-left px-4 py-2 text-sm font-semibold text-gray-600">Actions</th>
</tr></thead>
<tbody id="filelist"></tbody>
</table>
</div>

<script>
async function loadFiles() {
  const res = await fetch('/api/files');
  const files = await res.json();
  const tbody = document.getElementById('filelist');
  if (!files.length) { tbody.innerHTML = '<tr><td colspan=4 class=\'px-4 py-6 text-center text-gray-400\'>No files</td></tr>'; return; }
  tbody.innerHTML = files.map(f => \`
    <tr class="border-t hover:bg-gray-50">
      <td class="px-4 py-2"><a href="/files/\${f.name}" class="text-blue-500 underline hover:text-blue-700">\${esc(f.name)}</a></td>
      <td class="px-4 py-2 text-gray-500">\${f.size}</td>
      <td class="px-4 py-2 text-gray-500">\${f.modified}</td>
      <td class="px-4 py-2"><button onclick=\"delFile('\\${f.name}')\" class="text-red-500 hover:text-red-700">Delete</button></td>
    </tr>\`).join('');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function showStatus(msg, ok) {
  const el = document.getElementById('status');
  el.textContent = msg; el.className = 'mb-4 p-3 rounded-lg text-sm ' + (ok ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700');
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 4000);
}

async function handleFileSelect(e) {
  const file = e.target.files[0]; if (!file) return;
  await upload(file); e.target.value = '';
}

function handleDrop(e) {
  e.preventDefault(); e.currentTarget.classList.remove('border-blue-400','bg-blue-50');
  const file = e.dataTransfer.files[0]; if (!file) return;
  upload(file);
}

async function upload(file) {
  const fd = new FormData(); fd.append('file', file);
  try { const res = await fetch('/api/files/upload', {method:'POST', body:fd});
    if (!res.ok) { const j = await res.json(); showStatus(j.error||'Upload failed', false); return; }
    showStatus('Uploaded '+file.name, true); loadFiles();
  } catch(err) { showStatus('Upload error: '+err.message, false); }
}

async function delFile(name) {
  if (!confirm('Delete '+name+'?')) return;
  try { const res = await fetch('/api/files/'+encodeURIComponent(name), {method:'DELETE'});
    if (!res.ok) { const j = await res.json(); showStatus(j.error||'Delete failed', false); return; }
    showStatus('Deleted '+name, true); loadFiles();
  } catch(err) { showStatus('Delete error: '+err.message, false); }
}

loadFiles();
</script>
</body></html>"""


async def serve_file_ui(request: Request) -> Response:
    """Serve the file management UI."""
    return Response(content=FILE_UI_HTML, media_type="text/html")


def create_app(server: FastMCP) -> Starlette:
    """Create a Starlette app with SSE and Streamable HTTP transports.

    Exposes three MCP transport endpoints:
      GET  /sse        — SSE stream (legacy SSE transport)
      POST /messages/  — SSE message handler
      POST /mcp        — Streamable HTTP transport (recommended)

    Args:
        server: The MCP server instance.

    Returns:
        Starlette app serving both transports plus a /health endpoint.
    """
    if not settings.MCP_API_KEY:
        logger.warning("MCP_API_KEY is not set — auth is disabled")

    # SSE transport: GET /sse + POST /messages/
    sse_routes = list(server.sse_app().routes)

    # Streamable HTTP transport: POST /mcp  (also initialises the session manager)
    http_routes = list(server.streamable_http_app().routes)

    healthcheck_route = Route("/health", endpoint=healthcheck, methods=["GET"])
    # /files exact must come before /files/{filename} so Starlette doesn't greedily match
    file_ui_route = Route("/files", endpoint=serve_file_ui, methods=["GET"])
    file_route = Route("/files/{filename}", endpoint=serve_file, methods=["GET"])
    list_files_route = Route("/api/files", endpoint=list_files_api, methods=["GET"])
    upload_file_route = Route("/api/files/upload", endpoint=upload_file, methods=["POST"])
    delete_file_route = Route("/api/files/{filename}", endpoint=delete_file_api, methods=["DELETE"])

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        # StreamableHTTPSessionManager must be running while the server handles requests.
        async with server.session_manager.run():
            yield

    app = Starlette(
        debug=server.settings.debug,
        # Explicit routes first so they are matched before the MCP catch-all at "/"
        routes=[healthcheck_route, file_ui_route, file_route, list_files_route, upload_file_route, delete_file_route] + sse_routes + http_routes,
        lifespan=lifespan,
    )
    _cors_origins = [o.strip() for o in os.environ.get("MCP_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ApiKeyAuthMiddleware)
    return app


async def run_server(server: FastMCP) -> None:
    """Run the MCP server with SSE and Streamable HTTP transports.

    Args:
        server: The MCP server instance.
    """
    app = create_app(server)

    logger.info("Starting MCP server at %s:%s", settings.MCP_SERVER_HOST, settings.MCP_SERVER_PORT)
    logger.info("SSE transport:             GET  /sse")
    logger.info("SSE message handler:       POST /messages/")
    logger.info("Streamable HTTP transport: POST /  (recommended)")
    logger.info("Healthcheck:               GET  /health")
    logger.info("File UI:                   GET  /files")
    logger.info("File list:                 GET  /api/files")
    logger.info("File upload:               POST /api/files/upload")
    logger.info("File delete:               DEL  /api/files/{filename}")
    logger.info("File download:             GET  /files/{filename}")

    # Start the Starlette app directly with uvicorn so that the CORS middleware is applied
    import uvicorn
    config = uvicorn.Config(
        app,
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        log_level="info",
    )
    await uvicorn.Server(config).serve()


def main() -> None:
    """Start the MCP server."""
    server = create_server()
    register_tools(server)
    register_resources(server)
    asyncio.run(run_server(server))


if __name__ == "__main__":
    main()
