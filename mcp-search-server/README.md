# MCP Search Server

An advanced MCP (Model Context Protocol) server for semantic web search with browser automation. This server allows LLMs to perform web searches, fetch web pages, and extract content using browser automation.

## Features

- **Semantic Web Search**: Search using natural language queries via multiple search engines (DuckDuckGo, Bing, Google)
- **Browser Automation**: Playwright-based headless browser for rendering JavaScript-heavy pages
- **Content Extraction**: Extract structured content from web pages (headings, tables, links, images)
- **LLM-Friendly Output**: Formatted content optimized for LLM consumption
- **Docker-Ready**: Easy deployment with Docker Compose

## MCP Tools

### `search`
Search the web for information using the configured search engine.

**Parameters:**
- `query` (string): The search query
- `max_results` (integer, optional): Maximum number of results to return

**Returns:** JSON string of search results with title, URL, snippet, engine, and timestamp.

### `fetch`
Fetch and extract content from a URL using browser automation.

**Parameters:**
- `url` (string): The URL to fetch

**Returns:** Structured content including title, text content, headings, links, images, and tables.

### `deep_search`
Perform a web search and extract full content from top results.

**Parameters:**
- `query` (string): The search query
- `max_results` (integer, optional): Maximum number of results to return

**Returns:** Search results with extracted content from top 3 results.

## Setup

### Prerequisites

- Docker and Docker Compose
- (Optional) Search API keys for Bing or Google Custom Search

### Quick Start

1. Clone the repository and navigate to the project directory:
   ```bash
   cd mcp-search-server
   ```

2. Copy the environment file:
   ```bash
   cp .env.example .env
   ```

3. (Optional) Configure your search engine API keys in `.env`:
   ```bash
   # For Bing Search API
   SEARCH_API_KEY=your_bing_api_key
   
   # For Google Custom Search API
   SEARCH_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_cse_id
   ```

4. Build and start the container:
   ```bash
   docker compose up --build
   ```

### MCP Client Configuration

To connect your MCP client to this server, add the following configuration:

```json
{
  "mcpServers": {
    "search": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-search-server", "python", "-m", "src.server"]
    }
  }
}
```

Or for direct execution:

```json
{
  "mcpServers": {
    "search": {
      "command": "python",
      "args": ["-m", "src.server"]
    }
  }
}
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `MCP_SERVER_HOST` | Host address for the MCP server | `0.0.0.0` |
| `MCP_SERVER_PORT` | Port for the MCP server | `3100` |
| `SEARCH_ENGINE` | Search engine to use (`duckduckgo`, `bing`, `google`) | `duckduckgo` |
| `SEARCH_API_KEY` | API key for search engine (required for Bing/Google) | `` |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID (required for Google) | `` |
| `BROWSER_TIMEOUT` | Browser timeout in seconds | `30` |
| `MAX_RESULTS` | Maximum number of search results | `10` |
| `CACHE_ENABLED` | Enable caching (`true`/`false`) | `true` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `REDIS_HOST` | Redis host (optional, for distributed caching) | `` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_PASSWORD` | Redis password | `` |

## Project Structure

```
mcp-search-server/
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Multi-stage Docker build
├── .dockerignore           # Docker ignore patterns
├── .env.example            # Example environment variables
├── entrypoint.sh           # Container entrypoint script
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Python project configuration
├── src/
│   ├── __init__.py
│   ├── server.py           # Main MCP server
│   ├── config.py           # Configuration settings
│   ├── search/
│   │   ├── __init__.py
│   │   ├── engines.py      # Search engine implementations
│   │   └── models.py       # Data models
│   ├── browser/
│   │   ├── __init__.py
│   │   └── automation.py   # Playwright browser automation
│   ├── extractor/
│   │   ├── __init__.py
│   │   └── content.py      # Content extraction
│   └── tools/
│       ├── __init__.py
│       ├── search.py       # Search tool
│       ├── fetch.py        # Fetch tool
│       └── deep_search.py  # Deep search tool
└── tests/
    ├── __init__.py
    └── test_server.py
```

## Search Engine Options

### DuckDuckGo (Default)
- No API key required
- Free to use
- Limited rate limiting

### Bing Search API
- Requires API key from [Azure Portal](https://azure.microsoft.com/services/cognitive-services/bing-web-search-api/)
- More reliable for production use
- Rate limits based on subscription tier

### Google Custom Search API
- Requires API key and search engine ID from [Google Custom Search](https://programmablesearchengine.google.com/)
- 100 free searches per day
- Rate limits based on subscription

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Run the server
python -m src.server
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## License

MIT
