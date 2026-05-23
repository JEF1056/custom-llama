"""MCP Server for web search and browser automation."""

import asyncio
import contextlib
import json
import logging
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
        if path == "/mcp" or path.startswith("/files/") or path == "/messages/":
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
        streamable_http_path="/mcp",
        stateless_http=True,
        instructions=(
            "A generalized MCP tool server: web search, browser automation, HTTP requests, "
            "code execution, file operations, math, and time utilities.\n\n"
            "=== WHEN TO USE THIS SERVER ===\n"
            "1. When the agent is stuck or needs additional information from the web\n"
            "2. When the user specifies URLs that need to be fetched\n"
            "3. When the user asks for real-time data (news, stock prices, sports scores)\n"
            "4. When the user needs to verify content exists on a live website\n"
            "5. When the user needs to interact with a web application (fill forms, click buttons)\n"
            "6. When the user needs to make API calls to any REST endpoint\n"
            "7. When the user needs to run Python code for data processing or computation\n"
            "8. When the user needs mathematical computation (symbolic, numeric, or matrix)\n"
            "9. When the user needs to create, read, list, or delete files\n"
            "10. When the user needs current time or timezone conversion\n\n"
            "=== TOOL CATEGORIES ===\n"
            "1. Search Tools (search, fetch, deep_search): Web search and content extraction\n"
            "   - search: Search the web for information using a search engine\n"
            "   - fetch: Fetch and extract content from a specific URL\n"
            "   - deep_search: Search + extract full content from top 3 results\n\n"
            "2. HTTP Tools (http_request): Generic HTTP client for any REST API\n"
            "   - http_request: Make GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS requests\n"
            "   - Supports auth (basic, bearer, apikey), custom headers, body\n\n"
            "3. Code Execution (code_run): Sandboxed Python execution\n"
            "   - code_run: Execute Python code with numpy, pandas, sympy, scipy, etc.\n"
            "   - Blocked: os, subprocess, socket, threading, network modules\n\n"
            "4. Browser Automation Tools (browser_*): Full Playwright automation\n"
            "   - browser_navigate: Navigate to a URL\n"
            "   - browser_screenshot: Take a screenshot (returns as MCP image)\n"
            "   - browser_get_interactables: List all clickable/fillable elements with selectors\n"
            "   - browser_click: Click on an element\n"
            "   - browser_fill: Fill in a form field\n"
            "   - browser_evaluate: Execute JavaScript on the page\n"
            "   - browser_get_text: Get text content from an element\n"
            "   - browser_get_content: Get the full HTML content of the page\n"
            "   - browser_monitor: Monitor page changes over time\n"
            "   - browser_close: Close the browser session\n"
            "   - browser_list_sessions: List active browser sessions\n\n"
            "5. File Tools (create_file, file_upload, file_read, file_list, file_delete):\n"
            "   File I/O operations\n"
            "   - create_file: Create files in PDF/SVG/HTML/JSON/CSV/XML formats\n"
            "   - file_upload: Upload a file (base64 content) to the output directory\n"
            "   - file_read: Read file contents (text or base64 for binary)\n"
            "   - file_list: List files and directories\n"
            "   - file_delete: Delete a file\n"
            "   - Files accessible via download URL: FILE_BASE_URL/files/{filename}\n\n"
            "6. Math Tools (calculator): Advanced symbolic calculator\n"
            "   - calculator: Evaluate math expressions using SymPy\n"
            "   - Supports: arithmetic, trig, symbolic algebra, matrices, calculus, equation solving\n\n"
            "7. Time Tools (time_now): Time and timezone utilities\n"
            "   - time_now: Get current time in any timezone or convert between timezones\n\n"
            "8. Document Tools (xlsx_create, xlsx_read, xlsx_edit, pptx_create, pptx_read, pptx_edit, pptx_slide_image): Spreadsheet & Presentation I/O\n"
            "   - xlsx_create: Create Excel spreadsheets (formulas, formatting, charts)\n"
            "   - xlsx_read: Read existing spreadsheets and return structured JSON\n"
            "   - xlsx_edit: Edit existing spreadsheets (update cells, add rows, format, add sheets)\n"
            "   - pptx_create: Create PowerPoint presentations (text, images, charts, themes)\n"
            "   - pptx_read: Read existing presentations and return structured slide data\n"
            "   - pptx_edit: Edit existing presentations (add/update/delete slides, text, tables, notes)\n"
            "   - pptx_slide_image: Render a slide as a PNG image for visual inspection\n\n"
            "=== ITERATIVE DOCUMENT WORKFLOW ===\n"
            "The preferred approach for creating spreadsheets and presentations is iterative:\n"
            "1. Create a small initial version (e.g. headers + a few rows, or title + one slide)\n"
            "2. Read the file to verify structure (xlsx_read / pptx_read)\n"
            "3. Edit incrementally (xlsx_edit / pptx_edit) — add data, fix values, apply formatting\n"
            "4. For presentations, use pptx_slide_image to visually verify slide rendering\n"
            "5. Repeat steps 2-4 until the document is complete\n"
            "This approach is more reliable than trying to build everything in one create call.\n\n"
            "=== WORKFLOWS ===\n"
            "1. API + Data Processing: http_request to fetch API data -> code_run to process/analyze\n"
            "2. Search + Analysis: search -> fetch -> code_run to analyze the data\n"
            "3. Code + Files: code_run -> create_file/xlsx_create to save results\n"
            "4. Browser + HTTP: browser for JS-heavy pages, http_request for API endpoints\n"
            "5. Math + Code: calculator for symbolic math, code_run for numeric computation\n"
            "6. Data + Documents: code_run to compute -> xlsx_create to save as spreadsheet\n"
            "7. Presentations: code_run to generate charts -> pptx_create to embed as slides\n"
            "8. Iterative spreadsheets: xlsx_create (minimal) -> xlsx_edit (add data/format) -> xlsx_read (verify)\n"
            "9. Iterative presentations: pptx_create (title + 1 slide) -> pptx_edit (add slides) -> pptx_slide_image (verify)\n\n"
            "=== EXAMPLE SCENARIOS ===\n"
            "- Call an API: http_request(method='GET', url='...') -> parse JSON response\n"
            "- Analyze data: http_request to fetch -> code_run with pandas to analyze\n"
            "- Solve equations: calculator(expression='solve(x**2-4, x)')\n"
            "- Process web data: search -> fetch -> code_run to extract/transform\n"
            "- Generate files: code_run to compute -> create_file to save as CSV/JSON\n"
            "- Timezone work: time_now(timezone_name='Asia/Tokyo') or convert between zones\n"
            "- Multi-step: http_request -> code_run -> create_file -> file_read to verify\n"
            "- Spreadsheets: code_run with pandas -> xlsx_create to generate formatted report\n"
            "- Presentations: code_run with matplotlib -> pptx_create to build slide deck\n"
            "- Data pipeline: search -> fetch -> code_run -> xlsx_create -> xlsx_read to verify\n"
            "- Iterative spreadsheet: xlsx_create(headers) -> xlsx_edit(append_rows) -> xlsx_edit(format_range) -> xlsx_read to verify\n"
            "- Iterative presentation: pptx_create(title slide) -> pptx_edit(add_slide x3) -> pptx_slide_image(slide_index=1) to verify\n"
            "- Fix spreadsheet data: xlsx_read to inspect -> xlsx_edit(update_cell) to correct values\n"
            "- Refine presentation: pptx_read to inspect structure -> pptx_edit(update_slide_content) to fix text"),
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
        routes=sse_routes + http_routes + [healthcheck_route, file_ui_route, file_route, list_files_route, upload_file_route, delete_file_route],
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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
    logger.info("Streamable HTTP transport: POST /mcp  (recommended)")
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
