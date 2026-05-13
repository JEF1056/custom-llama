# MCP Search Server

An advanced MCP (Model Context Protocol) server for semantic web search with browser automation using SSE (Server-Sent Events) transport. This server allows LLMs to perform web searches, fetch web pages, and extract content using browser automation.

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

## Usage Pattern

### When to Use the MCP Server

The MCP search server should be called in the following scenarios:

1. **The agent is stuck or needs additional information**: When the agent encounters a knowledge gap or needs to gather real-time information to complete a task, it should use the search tools to find relevant data from the web.

2. **The user specifies URLs that need to be fetched**: When the user provides specific URLs and asks the agent to retrieve or analyze content from those pages, the agent should use the fetch tool or browser automation tools to access the content.

### Search Tools vs. Browser Automation Tools

The MCP server provides two categories of tools that work together:

| Category | Tools | Purpose |
|----------|-------|---------|
| **Search Tools** | `search`, `fetch`, `deep_search` | Quick information retrieval via HTTP requests and content extraction. Best for pages that don't require JavaScript rendering. |
| **Browser Automation Tools** | `browser_navigate`, `browser_screenshot`, `browser_click`, `browser_fill`, `browser_evaluate`, `browser_get_text`, `browser_get_content`, `browser_monitor`, `browser_close`, `browser_list_sessions` | Full browser interaction for JavaScript-heavy pages that can't be scraped with simple HTTP requests. |

**When to use search tools:**
- Quick web searches for information
- Fetching simple web pages (HTML without heavy JavaScript)
- Extracting structured content from pages that render server-side

**When to use browser automation tools:**
- Interacting with JavaScript-heavy pages (SPAs, dynamic content)
- Filling forms and clicking buttons
- Taking screenshots for visual verification
- Executing JavaScript on pages
- Monitoring page changes over time

### Workflow: Search + Browser Automation Together

The search tools and browser automation tools are designed to work together in a complementary workflow:

```
1. Initial Information Gathering
   └── Use search tools (search, fetch, deep_search) for quick results
       ├── search: Find relevant pages via web search
       ├── fetch: Extract content from specific URLs
       └── deep_search: Get search results with extracted content

2. Deep Dive into Specific Pages
   └── If search results indicate a page needs deeper interaction:
       ├── browser_navigate: Navigate to the URL
       ├── browser_screenshot: Verify the page loaded correctly
       ├── browser_get_content: Extract text content
       ├── browser_click: Click links/buttons to navigate
       ├── browser_fill: Fill forms (search boxes, login forms, etc.)
       ├── browser_evaluate: Execute JavaScript for custom data extraction
       └── browser_get_text: Get specific element text

3. Iterative Exploration
   └── Repeat step 2 as needed for different pages
       ├── Use browser_monitor for pages that change over time
       └── Use browser_close when done with a session
```

### Example Scenarios

#### Scenario 1: Researching a Topic

```
User: "Find me the latest information about quantum computing breakthroughs in 2026"

Agent workflow:
1. Call search(query="quantum computing breakthroughs 2026", max_results=5)
2. Review search results for relevant articles
3. Call fetch(url="https://example.com/quantum-breakthrough") for promising articles
4. If the page requires JavaScript rendering, use browser_navigate + browser_get_content
5. Compile findings from all sources
```

#### Scenario 2: User-Specified URL Analysis

```
User: "Check what's on this page and tell me about the latest features"
User provides: https://example.com/product

Agent workflow:
1. Call fetch(url="https://example.com/product")
2. If fetch returns empty or incomplete content (JS-heavy page):
   a. Call browser_navigate(url="https://example.com/product")
   b. Call browser_screenshot() to verify page loaded
   c. Call browser_get_content() to extract text
   d. Call browser_get_text(selector="h1") to get the main heading
3. Analyze and summarize the content
```

#### Scenario 3: Interactive Web Application

```
User: "Search for 'Python 3.13 release notes' and click on the first result"

Agent workflow:
1. Call search(query="Python 3.13 release notes", max_results=3)
2. Identify the first relevant URL from results
3. Call browser_navigate(url="https://docs.python.org/3/whatsnew/3.13.html")
4. Call browser_screenshot() to verify the page loaded
5. Call browser_click(selector="a.release-link") to click on a link
6. Call browser_get_content() to extract the new page content
```

#### Scenario 4: Form Interaction

```
User: "Go to https://example.com/search and search for 'AI trends'"

Agent workflow:
1. Call browser_navigate(url="https://example.com/search")
2. Call browser_screenshot() to verify the page loaded
3. Call browser_fill(selector="input.search-box", value="AI trends")
4. Call browser_click(selector="button.search-button")
5. Call browser_get_content() to extract search results
```

#### Scenario 5: Stuck Agent Recovery

```
User: "I need to know the current weather in Tokyo"

Agent workflow:
1. Call search(query="current weather Tokyo", max_results=3)
2. If search results don't provide detailed weather info:
   a. Call fetch(url="https://weather.com/tokyo")
   b. If fetch fails (JS-heavy page):
      - Call browser_navigate(url="https://weather.com/tokyo")
      - Call browser_get_content() to extract weather data
3. Compile and present the weather information
```

#### Scenario 6: Multi-Step Research with Monitoring

```
User: "Monitor this stock price page for the next minute and tell me if it changes"

Agent workflow:
1. Call browser_navigate(url="https://example.com/stock/ABC")
2. Call browser_screenshot() to verify the page loaded
3. Call browser_get_text(selector=".stock-price") to get initial price
4. Call browser_monitor(interval=10, duration=60) to capture changes
5. Compare the captured screenshots/text to identify price changes
```

#### Scenario 7: Deep Search with Fallback

```
User: "Find me documentation about React Server Components"

Agent workflow:
1. Call deep_search(query="React Server Components documentation", max_results=5)
2. Review extracted content from top results
3. If content is incomplete (e.g., requires JavaScript rendering):
   a. Call browser_navigate(url="https://react.dev/reference/rsc/server-components")
   b. Call browser_evaluate(script="JSON.stringify(document.querySelector('.content').innerHTML)")
   c. Extract and present the full documentation
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

To connect your MCP client to this server, add the following configuration. The server uses SSE (Server-Sent Events) transport on `/sse` and `/mcp` endpoints:

#### Docker Compose Configuration

```json
{
  "mcpServers": {
    "search": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-search-server", "python", "-m", "src.server"],
      "env": {
        "MCP_SERVER_HOST": "0.0.0.0",
        "MCP_SERVER_PORT": "3100"
      }
    }
  }
}
```

#### Direct Execution Configuration

```json
{
  "mcpServers": {
    "search": {
      "command": "python",
      "args": ["-m", "src.server"],
      "env": {
        "MCP_SERVER_HOST": "0.0.0.0",
        "MCP_SERVER_PORT": "3100"
      }
    }
  }
}
```

#### SSE Transport

The server uses the SSE (Server-Sent Events) transport protocol. The endpoints are:
- `GET /sse` - SSE endpoint for client to receive events (returns `text/event-stream`)
- `POST /mcp` - MCP JSON-RPC message endpoint (accepts `application/json`)

The server requires clients to accept both `application/json` and `text/event-stream` content types.

#### CORS Configuration

The server supports CORS to allow MCP clients from different origins. Configure the following environment variables:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `CORS_ALLOW_ALL` | Allow all CORS origins (use `*` for development) | `false` |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:8080,http://localhost:3000,http://localhost:2280` |

**Default CORS Origins for MCP Clients:**

| Client | Origin |
|--------|--------|
| llama.cpp Server Web UI | `http://localhost:8080` |
| Roo | `http://localhost:3000` |
| OpenCode | `http://localhost:2280` |

#### Full Configuration

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
| `CORS_ALLOW_ALL` | Allow all CORS origins (development only) | `false` |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:8080,http://localhost:3000,http://localhost:2280` |

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
