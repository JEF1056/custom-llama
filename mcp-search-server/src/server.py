"""MCP Server for web search and content extraction."""

import asyncio
import logging

from mcp.server import FastMCP

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
        instructions="A search server that can perform web searches, extract content from web pages, and automate browser interactions.",
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


async def run_server(server: FastMCP) -> None:
    """Run the MCP server with stdio transport.

    Args:
        server: The MCP server instance.
    """
    logger.info("Starting MCP server on stdio transport")
    await server.run_stdio_async()


def main() -> None:
    """Start the MCP server."""
    server = create_server()
    register_tools(server)
    asyncio.run(run_server(server))


if __name__ == "__main__":
    main()
