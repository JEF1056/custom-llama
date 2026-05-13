"""MCP Server for web search and content extraction using SSE and Streamable HTTP transports."""

import asyncio
import contextlib
import logging

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from src.config import settings
from src.tools.browser import browser_handler
from src.tools.deep_search import deep_search_handler
from src.tools.fetch import fetch_handler
from src.tools.search import search_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        instructions=(
            "A search server that can perform web searches, extract content from web pages, "
            "and automate browser interactions.\n\n"
            "=== WHEN TO USE THIS SERVER ===\n"
            "1. When the agent is stuck or needs additional information from the web\n"
            "2. When the user specifies URLs that need to be fetched\n"
            "3. When the user asks for real-time data (news, stock prices, weather, sports scores)\n"
            "4. When the user needs to verify content exists on a live website\n"
            "5. When the user needs to interact with a web application (fill forms, click buttons)\n"
            "6. When the user needs to monitor a page for changes over time\n"
            "7. When the user needs to check availability (product stock, appointment slots)\n"
            "8. When the user needs to track changes (price monitoring, competitor analysis)\n\n"
            "=== TOOL CATEGORIES ===\n"
            "1. Search Tools (search, fetch, deep_search): Use for quick HTTP-based retrieval\n"
            "   - search: Search the web for information using a search engine\n"
            "   - fetch: Fetch and extract content from a specific URL\n"
            "   - deep_search: Search + extract full content from top results\n\n"
            "2. Browser Automation Tools (browser_navigate, browser_screenshot, browser_click, etc.):\n"
            "   Use for JavaScript-heavy pages that require full browser interaction\n"
            "   - browser_navigate: Navigate to a URL\n"
            "   - browser_screenshot: Take a screenshot of the current page\n"
            "   - browser_click: Click on an element\n"
            "   - browser_fill: Fill in a form field\n"
            "   - browser_evaluate: Execute JavaScript on the page\n"
            "   - browser_get_text: Get text content from an element\n"
            "   - browser_get_content: Get the full HTML content of the page\n"
            "   - browser_monitor: Monitor page changes\n"
            "   - browser_close: Close the browser\n"
            "   - browser_list_sessions: List active browser sessions\n\n"
            "=== WORKFLOW: SEARCH + BROWSER AUTOMATION TOGETHER ===\n"
            "1. Initial information gathering via search tools (search -> fetch)\n"
            "2. Deep dive into specific pages via browser automation (navigate -> click -> extract)\n"
            "3. Iterative exploration based on findings\n\n"
            "=== USE CASE 1: GROUNDING/ADVISING LLM GENERATION ===\n"
            "When the LLM needs to generate responses based on real-time web data:\n"
            "- search for the topic -> fetch content from relevant URLs -> use extracted content to ground the response\n"
            "- deep_search for comprehensive information -> use extracted content to answer the question\n"
            "Example: \"What is the current weather in Tokyo?\" -> search -> fetch -> answer based on weather data\n\n"
            "=== USE CASE 2: REAL-TIME DATA FETCHING ===\n"
            "When the user needs live data from websites:\n"
            "- fetch specific URLs with live data (stock prices, sports scores, news)\n"
            "- browser_navigate to JS-heavy pages with live data -> browser_get_text to extract values\n"
            "Example: \"What's the price of Bitcoin?\" -> fetch -> extract price from page\n\n"
            "=== USE CASE 3: FORM SUBMISSION/AUTOMATION ===\n"
            "When the user needs to fill out forms or automate web workflows:\n"
            "- browser_navigate to the form page -> browser_fill to fill fields -> browser_click to submit\n"
            "- browser_get_text or browser_get_content to verify submission\n"
            "Example: \"Fill out this contact form\" -> navigate -> fill -> click -> verify\n\n"
            "=== USE CASE 4: AUTHENTICATION/SESSION MANAGEMENT ===\n"
            "When the user needs to access authenticated content:\n"
            "- browser_navigate to login page -> browser_fill to enter credentials -> browser_click to submit\n"
            "- browser_get_text to verify login success -> browser_navigate to protected page\n"
            "Example: \"Check my email\" -> navigate to login -> fill credentials -> access inbox\n\n"
            "=== USE CASE 5: DYNAMIC CONTENT EXTRACTION ===\n"
            "When the user needs data from SPAs that require JavaScript rendering:\n"
            "- browser_navigate to the page -> browser_get_text or browser_get_content to extract data\n"
            "- browser_evaluate to run custom JavaScript for data extraction\n"
            "Example: \"Get the latest reviews\" -> navigate -> extract reviews from SPA\n\n"
            "=== USE CASE 6: DATA COLLECTION/SCRAPING ===\n"
            "When the user needs to collect data from multiple pages:\n"
            "- search for the topic -> fetch multiple URLs -> extract data from each\n"
            "- browser_navigate to each page -> browser_get_text to collect data\n"
            "Example: \"Get all product prices from this site\" -> navigate -> extract -> repeat\n\n"
            "=== USE CASE 7: PROGRESS MONITORING ===\n"
            "When the user needs to monitor a page for changes over time:\n"
            "- browser_navigate to the page -> browser_monitor to capture periodic screenshots\n"
            "- Compare screenshots to detect changes\n"
            "Example: \"Monitor this product page for price drops\" -> navigate -> monitor -> compare\n\n"
            "=== USE CASE 8: INTERACTIVE DEBUGGING ===\n"
            "When the user needs to debug a web application:\n"
            "- browser_navigate to the page -> browser_screenshot to see the page\n"
            "- browser_evaluate to run debugging scripts -> browser_get_text to verify\n"
            "Example: \"Debug this form\" -> navigate -> screenshot -> evaluate -> fix\n\n"
            "=== USE CASE 9: AVAILABILITY CHECKING ===\n"
            "When the user needs to check product availability or appointment slots:\n"
            "- browser_navigate to the availability page -> browser_get_text to check status\n"
            "- browser_evaluate to check for availability indicators\n"
            "Example: \"Is this product in stock?\" -> navigate -> check availability\n\n"
            "=== USE CASE 10: COMPETITIVE INTELLIGENCE ===\n"
            "When the user needs to monitor competitor websites:\n"
            "- browser_navigate to competitor pages -> browser_screenshot to capture\n"
            "- browser_monitor for ongoing monitoring -> compare changes over time\n"
            "Example: \"What's my competitor's pricing?\" -> navigate -> extract -> compare\n\n"
            "=== EXAMPLE SCENARIOS ===\n"
            "- Researching a topic: search -> fetch -> browser fallback for JS-heavy pages\n"
            "- User-specified URL analysis: fetch -> browser fallback\n"
            "- Interactive web application: navigate -> click -> extract\n"
            "- Form interaction: navigate -> fill -> click -> extract\n"
            "- Stuck agent recovery: search -> fetch -> browser fallback\n"
            "- Multi-step research with monitoring: navigate -> monitor -> compare\n"
            "- Deep search with fallback: deep_search -> browser evaluate\n"
            "- Real-time data: fetch -> extract -> answer\n"
            "- Availability check: navigate -> get_text -> check status\n"
            "- Price monitoring: navigate -> monitor -> compare over time"
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


def healthcheck(request: Request) -> Response:
    """Simple healthcheck endpoint."""
    return Response(content="OK", media_type="text/plain", status_code=200)


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
    # SSE transport: GET /sse + POST /messages/
    sse_routes = list(server.sse_app().routes)

    # Streamable HTTP transport: POST /mcp  (also initialises the session manager)
    http_routes = list(server.streamable_http_app().routes)

    healthcheck_route = Route("/health", endpoint=healthcheck, methods=["GET"])

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        # StreamableHTTPSessionManager must be running while the server handles requests.
        async with server.session_manager.run():
            yield

    return Starlette(
        debug=server.settings.debug,
        routes=sse_routes + http_routes + [healthcheck_route],
        lifespan=lifespan,
    )


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
    asyncio.run(run_server(server))


if __name__ == "__main__":
    main()
