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

## Browser Automation Tools

The server provides 10 browser automation tools powered by Playwright for headless browser interaction:

### `browser_navigate`
Navigate to a URL.

**Parameters:**
- `url` (string): The URL to navigate to
- `wait_until` (string, optional): When to consider navigation completed (`load`, `domcontentloaded`, `networkidle`, `commit`)
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with page title, URL, and status.

### `browser_screenshot`
Take a screenshot of the current page.

**Parameters:**
- `full_page` (boolean, optional): Whether to capture the full page
- `path` (string, optional): The file path to save the screenshot. If None, saves to screenshot_dir.
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with screenshot path.

### `browser_click`
Click an element.

**Parameters:**
- `selector` (string): The CSS selector for the element
- `timeout` (integer, optional): Timeout in seconds. Defaults to BROWSER_TIMEOUT.
- `wait_until` (string, optional): When to consider navigation completed after click
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with success/failure status.

### `browser_fill`
Fill an input field.

**Parameters:**
- `selector` (string): The CSS selector for the input
- `value` (string): The value to fill
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with success/failure status.

### `browser_evaluate`
Execute JavaScript on the page.

**Parameters:**
- `script` (string): The JavaScript code to execute
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with the result of the JavaScript execution.

### `browser_get_text`
Get text content of an element.

**Parameters:**
- `selector` (string): The CSS selector for the element
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with the text content.

### `browser_get_content`
Get the page content (text extraction).

**Parameters:**
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with page text content and content length.

### `browser_monitor`
Periodic screenshot monitoring. Captures screenshots at regular intervals for a specified duration.

**Parameters:**
- `interval` (integer, optional): Seconds between screenshots (default: 5)
- `duration` (integer, optional): Total seconds to monitor (default: 30)
- `path` (string, optional): Output directory for screenshots. If None, uses screenshot_dir.
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with list of screenshot paths.

### `browser_close`
Close the browser session.

**Parameters:**
- `session_id` (string, optional): The session ID to close. If None, closes the default context.

**Returns:** JSON string with success/failure status.

### `browser_list_sessions`
List active browser sessions.

**Returns:** JSON string with list of session IDs and total count.

## Example Workflow

Here's an example workflow demonstrating how to use the browser automation tools together to interact with a web page:

```
1. Navigate to a URL:
   browser_navigate(url="https://example.com")

2. Take a screenshot to verify the page loaded:
   browser_screenshot()

3. Get the page content:
   browser_get_content()

4. Click a link:
   browser_click(selector="a.example-link")

5. Fill a form:
   browser_fill(selector="input#search", value="search term")

6. Execute JavaScript to extract data:
   browser_evaluate(script="document.querySelector('.data').textContent")

7. Get specific element text:
   browser_get_text(selector="h1")

8. Monitor page changes (captures screenshots every 5 seconds for 30 seconds):
   browser_monitor(interval=5, duration=30)

9. Close the browser session:
   browser_close()
```

## Setup

### Prerequisites

- Docker and Docker Compose
- (Optional) Search API keys for Bing or Google Custom Search

### Quick Start

1. Clone the repository and navigate to the project directory:
   ```bash
   cd ..
   ```

2. Copy the environment file:
   ```bash
   cp mcp-search-server/.env.example mcp-search-server/.env
   ```

3. (Optional) Configure your search engine API keys in `mcp-search-server/.env`:
   ```bash
   # For Bing Search API
   SEARCH_API_KEY=your_bing_api_key
   
   # For Google Custom Search API
   SEARCH_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_cse_id
   ```

4. Build and start the container:
   ```bash
   docker compose --profile mcp-search up --build
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
| `SCREENSHOT_DIR` | Directory to save screenshots | `/app/screenshots` |
| `MAX_RESULTS` | Maximum number of search results | `10` |
| `CACHE_ENABLED` | Enable caching (`true`/`false`) | `true` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `REDIS_HOST` | Redis host (optional, for distributed caching) | `` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_PASSWORD` | Redis password | `` |

## Project Structure

```
mcp-search-server/
├── Dockerfile              # Multi-stage Docker build
├── .dockerignore           # Docker ignore patterns
├── .env.example            # Example environment variables
├── entrypoint.sh           # Container entrypoint script
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Python project configuration
├── README.md               # This file
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
│       ├── browser.py      # Browser automation tools
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
