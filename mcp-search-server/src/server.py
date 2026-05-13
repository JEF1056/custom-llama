"""MCP Server for web search and content extraction."""

import asyncio
import logging
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.config import settings
from src.tools.deep_search import register_deep_search_tool
from src.tools.fetch import register_fetch_tool
from src.tools.search import register_search_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server):
    """Manage server startup and shutdown.

    Args:
        server: The MCP server instance.
    """
    logger.info("MCP Search Server starting up...")
    logger.info("Search engine: %s", settings.SEARCH_ENGINE)
    logger.info("Max results: %s", settings.MAX_RESULTS)
    logger.info("Cache enabled: %s", settings.CACHE_ENABLED)
    logger.info("Cache TTL: %s seconds", settings.CACHE_TTL)
    yield
    logger.info("MCP Search Server shutting down...")


def create_server() -> Server:
    """Create and configure the MCP server.

    Returns:
        Configured MCP server instance.
    """
    server = Server("mcp-search-server")
    server.lifespan = server_lifespan
    return server


def register_tools(server: Server) -> None:
    """Register all MCP tools with the server.

    Args:
        server: The MCP server instance.
    """
    register_search_tool(server)
    register_fetch_tool(server)
    register_deep_search_tool(server)


async def run_server(server: Server) -> None:
    """Run the MCP server with stdio transport.

    Args:
        server: The MCP server instance.
    """
    logger.info("Starting MCP server on stdio transport")
    async with stdio_server() as stream:
        await server.run(stream)


def main() -> None:
    """Start the MCP server."""
    server = create_server()
    register_tools(server)
    asyncio.run(run_server(server))


if __name__ == "__main__":
    main()
