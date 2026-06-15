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

Browser control is **programmatic**: instead of many narrow wrappers, the model
drives a Playwright browser directly by writing async Python. Three tools:

### `browser_run`
Run async Playwright Python against a live page. This is the primary browser tool.

The code body executes inside an async function with these names in scope:
- `page` — Playwright `Page` (`await page.goto(url)`, `page.click(sel)`, `page.fill(sel, val)`, …)
- `context` — the `BrowserContext` (new pages, cookies, …)
- `interactables()` — async helper returning clickable/fillable elements with selectors
- `mgr` — the `BrowserManager` (`mgr.get_content(page)`, `mgr.screenshot(page)`)

Use `return <value>` to send data back; `print()` output is also captured.

**Parameters:**
- `code` (string): async Playwright Python to execute.
- `session_id` (string, optional): stable name (e.g. `"main"`) to keep the page/cookies alive across calls. Omit for a one-off page closed after the call.
- `timeout` (integer, optional): seconds; defaults to `2 × BROWSER_TIMEOUT` (min 60).

**Returns:** JSON `{status, result, stdout, url, title, interactables, session_id}`.

> **Security:** `browser_run` executes Python **in-process** (full host/Python
> access), gated behind the server API key. Intended for trusted local use.

### `browser_screenshot`
Capture a screenshot and return it as an MCP image (for vision). A code return
value can't carry an image, so this stays a dedicated tool.

**Parameters:**
- `url` (string, optional): page to load first.
- `full_page` (boolean, optional): capture the full scrollable page.
- `session_id` (string, optional): screenshot a persistent session's current page.

**Returns:** MCP `ImageContent` (base64 PNG) + a `file://` resource URI.

### `browser_close`
Close a browser session and free its pages/cookies.

**Parameters:**
- `session_id` (string): the session id used with `browser_run`.

## Example Workflow

```python
# 1. Multi-step interaction in a persistent session via browser_run:
browser_run(session_id="main", code='''
    await page.goto("https://example.com")
    await page.fill("input#search", "playwright")
    await page.click("button[type='submit']")
    await page.wait_for_load_state("networkidle")
    return {
        "title": await page.title(),
        "heading": await page.inner_text("h1"),
        "links": await page.eval_on_selector_all("a", "els => els.map(e => e.href)"),
    }
''')

# 2. Visual check:
browser_screenshot(session_id="main")

# 3. Clean up:
browser_close(session_id="main")
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
| **Browser Automation** | `browser_run`, `browser_screenshot`, `browser_close` | Programmatic Playwright control for JavaScript-heavy pages. |
| **Data / Compute** | `code_run`, `time_now` | Sandboxed Python computation and time/timezone conversion. |
| **Output Paging** | `read_output` | Read the remainder of a large result that `fetch` / `deep_search` / `code_run` / `browser_run` only previewed. Those tools return a `*_handle` plus `*_next_offset`; pass them to `read_output` to read the full content in windows (like reading a file by line range). |

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
   └── code_run for computation

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
4. If the page requires JavaScript rendering, use browser_run to render + extract
5. Compile findings from all sources
```

#### Scenario 2: User-Specified URL Analysis

```
User: "Check what's on this page and tell me about the latest features"
User provides: https://example.com/product

Agent workflow:
1. Call fetch(url="https://example.com/product")
2. If fetch returns empty or incomplete content (JS-heavy page), use browser_run:
   browser_run(code='''
       await page.goto("https://example.com/product")
       return {"heading": await page.inner_text("h1"),
               "body": await mgr.get_content(page)}
   ''')
3. Analyze and summarize the content
```

#### Scenario 3: Interactive Web Application

```
User: "Search for 'Python 3.13 release notes' and click on the first result"

Agent workflow:
1. Call search(query="Python 3.13 release notes", max_results=3)
2. Drive the page with browser_run (persistent session):
   browser_run(session_id="main", code='''
       await page.goto("https://docs.python.org/3/whatsnew/3.13.html")
       await page.click("a.release-link")
       await page.wait_for_load_state("networkidle")
       return await mgr.get_content(page)
   ''')
3. browser_screenshot(session_id="main") to verify visually if needed
```

#### Scenario 4: Form Interaction

```
User: "Go to https://example.com/search and search for 'AI trends'"

Agent workflow:
1. browser_run(code='''
       await page.goto("https://example.com/search")
       await page.fill("input.search-box", "AI trends")
       await page.click("button.search-button")
       await page.wait_for_load_state("networkidle")
       return await mgr.get_content(page)
   ''')
```

#### Scenario 5: Stuck Agent Recovery

```
User: "I need to know the current weather in Tokyo"

Agent workflow:
1. Call search(query="current weather Tokyo", max_results=3)
2. If results are thin, fetch a weather page; if JS-heavy, fall back to browser_run
   to render and extract the live values
3. Compile and present the weather information
```

#### Scenario 6: Deep Search with Fallback

```
User: "Find me documentation about React Server Components"

Agent workflow:
1. Call deep_search(query="React Server Components documentation", max_results=5)
2. Review extracted content from top results
3. If content is incomplete (requires JavaScript rendering):
   browser_run(code='''
       await page.goto("https://react.dev/reference/rsc/server-components")
       return await page.inner_text(".content")
   ''')
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
│       ├── code_run.py     # code_run tool
│       ├── read_output.py  # read_output (large-output pagination)
│       └── time_now.py     # time_now tool
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
