# MCP Search Server

An advanced MCP (Model Context Protocol) server for semantic web search with browser automation using SSE (Server-Sent Events) transport. This server allows LLMs to perform web searches, fetch web pages, and extract content using browser automation.

## Features

- **Semantic Web Search**: Search using natural language queries via multiple search engines (DuckDuckGo, Bing, Google)
- **Browser Automation**: Playwright-based headless browser for rendering JavaScript-heavy pages
- **Content Extraction**: Extract structured content from web pages (headings, tables, links, images)
- **LLM-Friendly Output**: Formatted content optimized for LLM consumption
- **Docker-Ready**: Easy deployment with Docker Compose

## MCP Tools

### `advisor`
Ask the local LLM for expert reasoning on complex problems. Use this tool whenever
you need deeper analysis, when you're stuck on a reasoning task, or when you need
a second opinion on a plan.

**Parameters:**
- `context` (string): The problem context or background information (as much detail as needed).
- `question` (string): The specific question or task to ask the advisor.
- `model` (string, optional): The model to use (overrides config default).

**Returns:** JSON with status, model name, and the advisor's analysis.

> **When to use:** Whenever you're unsure about an approach, need to reason through
> a multi-step plan, or hit a dead end. Call it early and often.

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

The server provides 13 browser automation tools powered by Playwright for headless browser interaction:

### `browser_navigate`
Navigate to a URL.

**Parameters:**
- `url` (string): The URL to navigate to
- `wait_until` (string, optional): When to consider navigation completed (`load`, `domcontentloaded`, `networkidle`, `commit`)
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** JSON string with page title, URL, and status.

### `browser_screenshot`
Screenshot a URL or session page. Returns MCP ImageContent (base64 PNG) + resource URI.

One-off: provide url=. Session: provide session_id=.

**Parameters:**
- `url` (string, optional): The URL to screenshot
- `full_page` (boolean, optional): Whether to capture the full page
- `session_id` (string, optional): The session ID to use. If None, uses the default context.

**Returns:** Screenshot as base64 PNG (MCP ImageContent) plus a resource URI for later access.

### `browser_create_session`
Create a browser session for multi-step interactions.

Call first when performing multiple interactions on the same page. Pass the returned session_id to all subsequent browser tools. For one-off actions, use browser_navigate(url=...) directly.

**Returns:** JSON with connection status and default page info.

### `browser_get_interactables`
List all interactable elements (links, buttons, inputs) with CSS selectors + labels.

Use BEFORE clicking/filling to find the right selector. Requires session_id; use url= to navigate first.

**Parameters:**
- `url` (string, optional): The URL to load before getting interactables
- `session_id` (string, optional): The session ID to use

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

**Returns:** JSON string with list of screenshot resource URIs.

### `browser_close`
Close the browser session.

**Parameters:**
- `session_id` (string, optional): The session ID to close. If None, closes the default context.

**Returns:** JSON string with success/failure status.

### `browser_list_sessions`
List active browser sessions.

**Returns:** JSON string with list of session IDs and total count.

### `browser_connect`
Connect to a Chrome browser via CDP (Chrome DevTools Protocol).

The endpoint MUST be supplied by the user. After connecting, all browser tools operate on the user's real Chrome — visible on their screen, with their logins, extensions, and bookmarks intact.

**Parameters:**
- `endpoint` (string): The CDP endpoint URL (ws://, wss://, http://, or https://)

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

3. **Complex reasoning tasks**: When faced with multi-step planning, debugging, or analysis, call the `advisor` tool to get expert reasoning from the local LLM before proceeding.

### Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| **Expert Reasoning** | `advisor` | Call this whenever you're stuck, need a second opinion, or face a complex reasoning task. Use it liberally. |
| **Search Tools** | `search`, `fetch`, `deep_search` | Quick information retrieval via HTTP requests and content extraction. |
| **Browser Automation** | `browser_navigate`, `browser_screenshot`, `browser_click`, `browser_fill`, `browser_evaluate`, `browser_get_text`, `browser_get_content`, `browser_monitor`, `browser_close`, `browser_list_sessions`, `browser_create_session`, `browser_get_interactables`, `browser_connect` | Full browser interaction for JavaScript-heavy pages. |
| **Data / Files** | `code_run`, `calculator`, `xlsx_create`, `xlsx_read`, `xlsx_edit`, `pptx_create`, `pptx_edit`, `pptx_read`, `pptx_slide_image`, `create_file`, `file_read`, `file_list`, `file_delete`, `file_upload`, `http_request`, `time_now` | Computation, file I/O, spreadsheet/presentation generation, HTTP calls, time zones. |

### Workflow: Search + Reasoning + Browser

```
1. Understand the problem
   └── If complex or multi-step: call advisor(context, question) first

2. Gather information
   ├── search/query: Find relevant pages
   ├── fetch: Extract content from specific URLs
   └── deep_search: Get results with extracted content

3. Reason and plan
   └── Call advisor() to validate approach or get guidance

4. Execute
   ├── Browser tools for JS-heavy pages
   ├── code_run for computation
   ├── file operations for output
   └── http_request for API calls

5. Verify and iterate
   └── Use advisor() again if results seem off or you need a different approach
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

2. The server reads configuration from environment variables. Set them in the `docker-compose.yml` or your `.env` file.

3. (Optional) Configure your search engine API keys:
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

To connect your MCP client to this server, add the following configuration. The server uses SSE and Streamable HTTP transports on `/sse` and `/` endpoints:

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

The server uses both SSE and Streamable HTTP transport. The endpoints are:
- `GET /sse` — SSE endpoint for receiving events (returns `text/event-stream`)
- `POST /messages/` — SSE message endpoint (accepts `application/json`)
- `POST /` — Streamable HTTP transport endpoint (accepts `application/json`)

#### CORS Configuration

The server supports CORS to allow MCP clients from different origins. Configure the following environment variable:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `MCP_CORS_ORIGINS` | Comma-separated list of allowed CORS origins (use `*` for all origins) | `http://localhost:8080,http://localhost:3000,http://localhost:2280` |

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
| `SCREENSHOT_DIR` | Directory to save screenshots | `/app/mcp-files/screenshots` |
| `MAX_RESULTS` | Maximum number of search results | `10` |
| `CACHE_ENABLED` | Enable caching (`true`/`false`) | `true` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `REDIS_HOST` | Redis host (defined but not currently used) | `` |
| `REDIS_PORT` | Redis port (defined but not currently used) | `6379` |
| `REDIS_PASSWORD` | Redis password (defined but not currently used) | `` |
| `MCP_CORS_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:8080,http://localhost:3000,http://localhost:2280` |

## Project Structure

```
mcp-search-server/
├── Dockerfile              # Multi-stage Docker build
├── .dockerignore           # Docker ignore patterns
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
│       ├── advisor.py      # Expert reasoning via local LLM
│       ├── browser.py      # Browser automation tools
│       ├── search.py       # Search tool
│       ├── fetch.py        # Fetch tool
│       ├── deep_search.py  # Deep search tool
│       ├── filetool.py     # create_file tool
│       ├── file_ops.py     # file_read, file_list, file_delete, file_upload tools
│       ├── http_request.py # http_request tool
│       ├── calculator.py   # calculator tool
│       ├── code_run.py     # code_run tool
│       ├── time_now.py     # time_now tool
│       ├── xlsx_create.py  # xlsx_create tool
│       ├── xlsx_edit.py    # xlsx_edit tool
│       ├── xlsx_read.py    # xlsx_read tool
│       ├── pptx_create.py  # pptx_create tool
│       ├── pptx_edit.py    # pptx_edit tool
│       ├── pptx_read.py    # pptx_read tool
│       └── pptx_slide_image.py  # pptx_slide_image tool
└── tests/
    ├── __init__.py
    ├── test_advisor.py     # Advisor tool tests
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
