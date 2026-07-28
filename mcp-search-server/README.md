# MCP Search Server

An advanced MCP (Model Context Protocol) server for semantic web search with browser automation using SSE (Server-Sent Events) transport. This server allows LLMs to perform web searches, fetch web pages, and extract content using browser automation.

## Features

- **Semantic Web Search**: Search using natural language queries via multiple search engines (DuckDuckGo, Bing, Google)
- **Browser Automation**: Fine-grained, discrete Playwright tools with anti-detection (patchright + real Chrome, headful via Xvfb, trusted input pipeline)
- **Content Extraction**: Extract structured content from web pages (headings, tables, links, images) with token-aware summarization
- **LLM-Friendly Output**: Formatted content optimized for LLM consumption, with pagination for large outputs
- **Docker-Ready**: Easy deployment with Docker Compose

## MCP Tools

### `advisor`
Ask the local LLM for expert reasoning on complex problems. Use this tool whenever
you need deeper analysis, when you're stuck on a reasoning task, or when you need
a second opinion on a plan.

**Parameters:**
- `context` (string): All relevant background for the question — be generous; the advisor only sees what you pass.
- `question` (string): The specific question or task to reason about.

**Returns:** Markdown — the advisor's response under a header naming the model.

> **When to use:** Whenever you're unsure about an approach, need to reason through
> a multi-step plan, or hit a dead end. Call it early and often.

### `search`
Search the web for information using the configured search engine.

**Parameters:**
- `query` (string): What to search the web for.
- `max_results` (integer, optional): Max results to return; defaults to the server config (typically 10).

**Returns:** Markdown — a numbered list of results (title, URL, snippet).

### `fetch`
Fetch and extract text from a URL (renders JS via headless browser).

**Parameters:**
- `url` (string): The page URL to fetch and extract text from.
- `truncate` (string, optional): How to trim long pages: "always" (default) | "never" | "main_only" | "code_only".
- `code_block_max_chars` (integer, optional): Override the per-code-block character limit.
- `sections` (list of strings, optional): Heading texts to extract only those sections (useful for long pages).

**Returns:** Markdown — title, URL, the extracted content, and (when the page
was truncated) a footer with the `read_output` handle to fetch the rest.

### `deep_search`
Search the web and extract full page content from the top 3 results in one call.

**Parameters:**
- `query` (string): What to search the web for.
- `max_results` (integer, optional): Size of the search pool; full content is extracted from the top 3 only.

**Returns:** Markdown — the top results with their extracted content, followed by
the remaining search hits as a link list.

## Browser Automation Tools

Browser control uses **discrete, fine-grained tools** rather than a single wrapper.
Each tool handles one specific browser action, reducing the chance of errors and
making it easier for the model to reason about what it's doing.

### `browser_screenshot`
Capture a screenshot and return it as an MCP image (for vision).

**Parameters:**
- `url` (string, optional): Optional page to load before capturing.
- `full_page` (boolean, optional): Capture the entire scrollable page instead of just the viewport.
- `session_id` (string, optional): Screenshot the current page of a persistent session.

**Returns:** MCP `ImageContent` (base64 PNG) + a `file://` resource URI.

### `navigate_page`
Navigate to a URL and return page state.

**Parameters:**
- `url` (string): The URL to navigate to.
- `session_id` (string, optional): Persistent session id. Omit for one-off navigation.
- `wait_until` (string, optional): When to consider navigation complete. Default: domcontentloaded.

**Returns:** JSON with status, url, title, accessibility snapshot, and interactables count.

### `take_snapshot`
Take an ARIA accessibility snapshot of the current page.

**Parameters:**
- `session_id` (string, optional): Persistent session id. Omit for one-off.
- `depth` (integer, optional): Maximum ARIA tree depth. None for full. Recommended: 3 for page recon.

**Returns:** JSON with status and the accessibility snapshot (with `[ref=eN]` markers for element targeting).

### `page_state`
Get unified page state: URL, title, accessibility snapshot, interactables, scroll.

**Parameters:**
- `session_id` (string, optional): Persistent session id. Omit for one-off.

**Returns:** JSON with status, url, title, accessibility, interactables_count, scroll_position, and headings.

### `click`
Click an element on the current page by CSS selector.

**Parameters:**
- `selector` (string): CSS selector for the element to click. Use `get_interactables()` to discover selectors first.
- `session_id` (string, optional): Persistent session id. Omit for one-off.

**Returns:** JSON with status, action, selector, url, title, and interactables_count.

### `fill`
Fill an input field on the current page by CSS selector.

**Parameters:**
- `selector` (string): CSS selector for the input element to fill.
- `value` (string): The value to fill into the input.
- `session_id` (string, optional): Persistent session id. Omit for one-off.

**Returns:** JSON with status, action, selector, value, url, and title.

### `get_text`
Get the inner text of an element by CSS selector.

**Parameters:**
- `selector` (string): CSS selector for the element whose text to read.
- `session_id` (string, optional): Persistent session id. Omit for one-off.

**Returns:** JSON with status and the element's text content.

### `evaluate`
Evaluate JavaScript on the current page.

**Parameters:**
- `script` (string): JavaScript code to evaluate on the page.
- `session_id` (string, optional): Persistent session id. Omit for one-off.

**Returns:** JSON with status and the JavaScript result.

### `get_interactables`
Get all clickable and fillable elements on the current page.

**Parameters:**
- `session_id` (string, optional): Persistent session id. Omit for one-off.

**Returns:** JSON with status and a list of elements (index, tag, type, text, name, id, placeholder, selector, visible).

### `get_content`
Get the rendered page content as markdown text.

**Parameters:**
- `session_id` (string, optional): Persistent session id. Omit for one-off.

**Returns:** JSON with status and the page content converted to markdown via html2text.

### `browser_close`
Close a browser session and free its pages/cookies.

**Parameters:**
- `session_id` (string): The session id used with browser tools.

**Returns:** JSON with status and a confirmation message.

> **Anti-detection strategy:** The server uses **patchright** (a CDP-patched Playwright that closes protocol leaks), **real Google Chrome** (not bundled Chromium, which exposes a "HeadlessChrome" brand flag), and **headful mode via Xvfb** when available (a real windowed browser is far harder to fingerprint). No fingerprint injection, no spoofed user-agents, no stealth plugins — the real Chrome identity stays consistent with the host's IP and timezone. All input (navigation, clicks, fills) uses the real CDP pipeline so events carry `isTrusted === true`.
- Default timeout is `2 × BROWSER_TIMEOUT` (min 60s); raise it with `timeout=`.
- **Need visual context?** Text alone misses a lot — take a `browser_screenshot` (same `session_id`) when you need to see page layout, rendered state, an image/chart, confirm a click/scroll worked, or see what's blocking you (modal, cookie banner, captcha, login wall).

**Parameters:**
- `code` (string): async Playwright Python to execute.
- `session_id` (string, optional): stable name to keep the page/cookies alive across calls. Make it unique (topic word + a few random chars, e.g. `"walmart-7q3"`) to avoid colliding with another task's session; reuse the same id on follow-ups. Omit for a one-off page closed after the call.
- `timeout` (integer, optional): seconds; defaults to `2 × BROWSER_TIMEOUT` (min 60).

**Returns:** markdown — a status/session line, the page title+URL, the returned
result, stdout, and the visible interactables. **On error** the same report is
returned with `status: error` plus an **Error** block (traceback with the failing
code line), a corrective hint, the page URL/title reached so far, any stdout
printed before the failure, and the interactables list — so you can see where you
were and fix the selector/step and retry.

> **Security:** Browser tools execute Python **in-process** (full host/Python
> access), gated behind the server API key. Intended for trusted local use.

### `browser_screenshot`
Capture a screenshot and return it as an MCP image (for vision). A code return
value can't carry an image, so this stays a dedicated tool. Reach for it whenever
you need **visual context** — page layout, rendered state, an image/chart,
confirming an action worked, or seeing what's on a page the text didn't capture
(modal, captcha, login wall).

**Parameters:**
- `url` (string, optional): page to load first.
- `full_page` (boolean, optional): capture the full scrollable page.
- `session_id` (string, optional): screenshot a persistent session's current page.

**Returns:** MCP `ImageContent` (base64 PNG) + a `file://` resource URI.

### `browser_sessions`
List the live browser sessions so you can reuse one instead of guessing ids.

**Parameters:** none.

**Returns:** JSON with a `sessions` array; each entry has `session_id`,
`current_url`, `current_title`, `pages` (open page count), and `idle_seconds`.

### `browser_close`
Close a browser session and free its pages/cookies.

**Parameters:**
- `session_id` (string): the session id used with browser tools.

**Returns:** JSON with status and a confirmation message.

## Example Workflow

```python
# 1. Navigate and inspect the page:
navigate_page(url="https://example.com", session_id="search-9k2")

# 2. Discover clickable/fillable elements:
get_interactables(session_id="search-9k2")

# 3. Fill and click using known selectors:
fill(selector="input#search", value="playwright", session_id="search-9k2")
click(selector="button[type='submit']", session_id="search-9k2")

# 4. Get the rendered page content:
get_content(session_id="search-9k2")

# 5. Visual check (if needed):
browser_screenshot(session_id="search-9k2")

# 6. Clean up:
browser_close(session_id="search-9k2")
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
| **Browser Navigation** | `navigate_page`, `take_snapshot`, `page_state` | Navigate pages and get accessibility/structural context. |
| **Browser Interaction** | `click`, `fill`, `get_text`, `evaluate` | Interact with page elements using CSS selectors. |
| **Browser Discovery** | `get_interactables`, `get_content` | Discover clickable/fillable elements and get rendered page text. |
| **Visual Context** | `browser_screenshot`, `browser_close` | Capture screenshots and clean up sessions. |
| **Data / Compute** | `code_run`, `time_now` | Sandboxed Python computation and time/timezone conversion. |
| **Output Paging** | `read_output` | Read the remainder of a large result that `fetch` / `deep_search` / `code_run` / `take_snapshot` / `get_content` only previewed. Those tools return a `*_handle` plus `*_next_offset`; pass them to `read_output` to read the full content in windows (like reading a file by line range). |

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
4. If the page requires JavaScript rendering, use navigate_page + get_content
5. Compile findings from all sources
```

#### Scenario 2: User-Specified URL Analysis

```
User: "Check what's on this page and tell me about the latest features"
User provides: https://example.com/product

Agent workflow:
1. Call fetch(url="https://example.com/product")
2. If fetch returns empty or incomplete content (JS-heavy page), use navigate_page + get_content:
    navigate_page(url="https://example.com/product", session_id="product-xk")
    get_content(session_id="product-xk")
3. Analyze and summarize the content
```

#### Scenario 3: Interactive Web Application

```
User: "Search for 'Python 3.13 release notes' and click on the first result"

Agent workflow:
1. Call search(query="Python 3.13 release notes", max_results=3)
2. Drive the page with browser tools (persistent session, unique id):
    navigate_page(url="https://docs.python.org/3/whatsnew/3.13.html", session_id="pydocs-5m")
    get_interactables(session_id="pydocs-5m")
    click(selector="a.release-link", session_id="pydocs-5m")
    get_content(session_id="pydocs-5m")
3. browser_screenshot(session_id="pydocs-5m") to verify visually if needed
```

#### Scenario 4: Form Interaction

```
User: "Go to https://example.com/search and search for 'AI trends'"

Agent workflow:
1. navigate_page(url="https://example.com/search", session_id="form-xk")
    fill(selector="input.search-box", value="AI trends", session_id="form-xk")
    click(selector="button.search-button", session_id="form-xk")
    get_content(session_id="form-xk")
```

#### Scenario 5: Stuck Agent Recovery

```
User: "I need to know the current weather in Tokyo"

Agent workflow:
1. Call search(query="current weather Tokyo", max_results=3)
2. If results are thin, fetch a weather page; if JS-heavy, fall back to navigate_page + get_content
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
navigate_page(url="https://react.dev/reference/rsc/server-components", session_id="react-xk")
    get_content(session_id="react-xk")
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
