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
from src.tools.advisor import advisor_handler
from src.tools.browser import browser_handler
from src.tools.code_run import code_run_handler
from src.tools.deep_search import deep_search_handler
from src.tools.fetch import fetch_handler
from src.tools.read_output import read_output_handler
from src.tools.search import search_handler
from src.tools.time_now import time_now_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware with per-IP rate limiting on failures.

    Reads MCP_API_KEY from settings.  When the key is empty, auth is skipped
    entirely (backward-compatible).  When set, requires
    `Authorization: Bearer <key>` on /mcp and /messages/ endpoints.
    The /health, GET /sse, and /files/* endpoints bypass auth so that embedded
    markdown images can be fetched by browsers without credentials.
    The /api/files management endpoints remain protected.

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

        # Skip CORS preflight requests — browsers don't send auth headers on OPTIONS
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.scope["path"]

        # Public endpoints — no auth
        if path == "/health":
            return await call_next(request)
        if path == "/sse" and request.method == "GET":
            return await call_next(request)
        # /files/* is public so embedded markdown images render without credentials.
        if path.startswith("/files/"):
            return await call_next(request)

        # Protected endpoints
        if path == "/" or path == "/messages/" or path.startswith("/api/files"):
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
        # Stateful (session-backed) Streamable HTTP. Stateless mode cannot stream
        # server->client progress notifications back during a tool call, so the
        # UI never receives ctx.report_progress() updates. A persistent session
        # keeps the per-request SSE response stream open long enough to deliver
        # them. Safe here: a single server replica, and the UI auto-reconnects on
        # session expiry (e.g. after a restart).
        stateless_http=False,
        instructions=(
            "Tools: web search, browser automation (discrete tools), "
            "page fetch, Python execution, time, and an advisor LLM for reasoning.\n\n"
            "USE TOOLS PROACTIVELY. Do not answer from memory when a tool can verify. "
            "Search/fetch before stating any fact that may be outdated. Run code instead "
            "of doing math in your head. Ask the advisor when reasoning gets hard. "
            "Chain tools freely; a wrong guess costs more than a tool call.\n\n"
            "Which tool when:\n"
            "- Need a fact / find sources -> search (titles+snippets).\n"
            "- Read one known page -> fetch.\n"
            "- Research a topic across sources -> deep_search (search + extract top 3).\n"
            "- Navigate a page -> navigate_page(url).\n"
            "- Understand page structure -> take_snapshot() for ARIA accessibility tree, "
            "or page_state() for unified context.\n"
            "- Interact with a page -> click(selector), fill(selector, value), "
            "get_text(selector), evaluate(script).\n"
            "- Discover what's clickable -> get_interactables() then use returned selectors.\n"
            "- See a page visually (layout, chart, confirm an action) -> browser_screenshot.\n"
            "- Math / parse / transform data (no web) -> code_run.\n"
            "- Hard reasoning or design decision -> advisor.\n"
            "- Current time / timezone -> time_now.\n"
            "- Continue a previewed long result -> read_output.\n\n"
            "Output format: all tools return compact text. Tool results use `---` as a "
            "separator between content and metadata footers (pagination hints, interactables "
            "counts, etc.). Pagination hints show `read_output(handle=\"...\", offset=N)` "
            "calls you can follow directly.\n\n"
            "Tool guide:\n"
            "- advisor(context, question): local reasoning LLM. Call early on complex tasks. "
            "Returns: `[advisor: model]` header + response.\n"
            "- search(query): fast titles+snippets. Returns: `Search: \"...\" — N results` "
            "header + numbered list.\n"
            "- fetch(url): full page text. Returns: `[Title](url)` header + content + "
            "pagination footer if truncated.\n"
            "- deep_search(query): search + extract top results in one call.\n"
            "- Browser workflow: navigate_page(url) → take_snapshot() to understand "
            "the page → click/fill/get_text to interact → take_snapshot() to confirm. "
            "Use get_interactables() to discover clickable elements before acting. "
            "Use page_state() for unified context. Use evaluate(script) to run JavaScript. "
            "Use get_content() for rendered markdown text.\n"
            "- browser_screenshot(url, full_page): capture visual context. "
            "Use when you need to see layout, charts, or confirm an action worked.\n"
            "- code_run(code): sandboxed Python for math, data, parsing.\n"
            "- time_now: current time / timezone conversion (default PST).\n"
            "- read_output(handle, offset): paginate through large outputs. "
            "Follow the `read_output(handle=\"...\", offset=N)` in footers. "
            "Keep calling until the footer shows `End:`.\n\n"
            "Typical chain: advisor → search/fetch (→ read_output if previewed) → code_run → answer."
        ),
    )
    return server


def register_tools(server: FastMCP) -> None:
    """Register all MCP tools with the server.

    Args:
        server: The MCP server instance.
    """
    advisor_handler(server)
    search_handler(server)
    fetch_handler(server)
    deep_search_handler(server)
    browser_handler(server)
    code_run_handler(server)
    time_now_handler(server)
    read_output_handler(server)


def register_resources(server: FastMCP) -> None:
    """Register MCP Resources and ResourceTemplates.

    Registers a resource template for reading files in the output directory
    (e.g. screenshots saved by browser_screenshot).
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

    Exposes files produced by MCP tools (e.g. browser_screenshot) as
    downloadable resources via the FILE_BASE_URL. Supports nested paths
    (e.g. ``screenshots/foo.png``) while rejecting path traversal so a
    request can never escape FILE_OUTPUT_DIR.
    """
    filename = request.path_params["filename"]
    base_dir = Path(settings.FILE_OUTPUT_DIR).resolve()
    file_path = (base_dir / filename).resolve()
    # Reject any path that escapes the output directory (OWASP A01).
    if base_dir != file_path and base_dir not in file_path.parents:
        return Response(content="Not found", status_code=404)
    if not file_path.exists() or not file_path.is_file():
        return Response(content="Not found", status_code=404)
    return FileResponse(
        str(file_path),
        headers={"Cross-Origin-Resource-Policy": "cross-origin"},
    )


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


FILE_UI_HTML = r"""<!DOCTYPE html>
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
    file_route = Route("/files/{filename:path}", endpoint=serve_file, methods=["GET"])
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
    # Auth first (inner), then CORS (outer) so CORS headers appear on all responses
    app.add_middleware(ApiKeyAuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # Stateful Streamable HTTP returns the session id (and negotiated protocol
        # version) as RESPONSE headers. A cross-origin browser client (the WebUI on
        # chat.jessfan.com calling mcp.jessfan.com) can only read these via JS if
        # they are listed here; otherwise the client can't echo mcp-session-id on
        # follow-up requests and the server rejects them with "Missing session ID".
        expose_headers=["mcp-session-id", "mcp-protocol-version"],
    )
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
