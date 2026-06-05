"""Benchmark scenario definitions — prompts and tool schemas for each workload type.

Each scenario is designed to reveal throughput differences for specific config flags:
  - general_text: fluent prose, low ngram repeat → baseline decode speed
  - coding: repetitive syntax → ngram speculation advantage
  - agentic: mixed structured reasoning → MTP vs ngram tradeoff
  - instruction_following: tables/lists → DRY relevance, structured output
  - tool_calling: JSON tool_calls → ngram excels at templates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """A benchmark scenario with system prompt, user prompt, and optional tools."""
    name: str
    system: str
    user: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS_WEATHER: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "Convert a temperature value between Fahrenheit and Celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Temperature value"},
                    "from_unit": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                    },
                    "to_unit": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": "Search for events in a city within a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "category": {
                        "type": "string",
                        "enum": ["outdoor", "indoor", "music", "sports", "food"],
                        "description": "Event category filter",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format",
                    },
                },
                "required": ["city"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="general_text",
        description="Fluent prose — raw decode speed, low ngram repeat",
        system="You are a thoughtful essayist who writes in clear, engaging prose. Always write at maximum length.",
        user=(
            "Write a detailed essay analyzing how artificial intelligence is transforming "
            "modern education. Structure your essay with:\n"
            "- An introduction that sets the historical context (at least 3 paragraphs)\n"
            "- Section 1: Benefits of AI in education — personalized learning, accessibility, "
            "automated grading, intelligent tutoring systems (at least 4 paragraphs)\n"
            "- Section 2: Risks and challenges — academic integrity, bias in AI systems, "
            "digital divide, teacher displacement, data privacy (at least 4 paragraphs)\n"
            "- Section 3: Policy recommendations — regulation frameworks, teacher training, "
            "equity measures, international cooperation (at least 4 paragraphs)\n"
            "- Section 4: Case studies — discuss at least 3 hypothetical implementations "
            "in different countries with specific outcomes and statistics (at least 3 paragraphs)\n"
            "- A conclusion with future outlook (at least 2 paragraphs)\n\n"
            "Use specific examples and cite hypothetical studies to support each point. "
            "Do not use bullet points or numbered lists — write in flowing paragraph form only. "
            "Each paragraph should be at least 4 sentences long."
        ),
    ),
    Scenario(
        name="coding",
        description="Repetitive syntax — ngram speculation advantage",
        system="You are an expert Python developer. Write clean, production-ready code with comprehensive type hints and docstrings.",
        user=(
            "Write a Python module that implements a complete `TaskScheduler` system with the "
            "following features:\n"
            "1. A `Task` dataclass with name, priority (int), async callable, metadata dict, "
            "created_at timestamp, and tags list.\n"
            "2. A `TaskResult` dataclass with task name, success/failure, duration, retry "
            "count, error message, start/end timestamps, and output value.\n"
            "3. A `TaskScheduler` class with:\n"
            "   a. Priority queue backed by `heapq`\n"
            "   b. Delayed execution with specified delay in seconds\n"
            "   c. Retry logic up to N times with exponential backoff (base 2 seconds)\n"
            "   d. Configurable worker pool using `asyncio`\n"
            "   e. Graceful shutdown: cancel pending, wait for running\n"
            "   f. Task dependency graph — tasks can depend on other tasks\n"
            "   g. Event hooks: on_task_start, on_task_complete, on_task_failure\n"
            "   h. Statistics tracking: total runs, success rate, avg duration per task\n"
            "4. A `TaskMonitor` class that logs task progress and can generate a summary "
            "report as a formatted string.\n"
            "5. A comprehensive `if __name__` demo that:\n"
            "   a. Creates 10 example tasks with various priorities and dependencies\n"
            "   b. Includes tasks that deliberately fail to test retry logic\n"
            "   c. Uses event hooks to print progress\n"
            "   d. Prints the final statistics report\n\n"
            "Write the complete module with all imports, classes, and full implementation. "
            "Include docstrings for every class and method."
        ),
    ),
    Scenario(
        name="agentic",
        description="Multi-step reasoning — tests mixed content decode",
        system=(
            "You are an AI research agent. You have access to the following tools:\n"
            "- web_search(query: str) -> list of search results with titles and snippets\n"
            "- file_read(path: str) -> file contents as string\n"
            "- code_execute(language: str, code: str) -> execution output\n"
            "- database_query(sql: str) -> query results as JSON\n\n"
            "For each step, show your detailed reasoning (at least 3-4 sentences), "
            "which tool you would call and why, the exact tool call with arguments, "
            "a hypothetical response, and your analysis of that response. "
            "Be extremely thorough and methodical."
        ),
        user=(
            "I need you to research and compile a comprehensive performance comparison of "
            "LLM inference engines on consumer GPUs (RTX 3090, 4090, 4080). Specifically:\n"
            "1. Find the latest benchmarks for vLLM, llama.cpp, TensorRT-LLM, SGLang, and "
            "ExLlamaV2 — search for each individually.\n"
            "2. Compare tokens/second for 7B, 13B, 27B, and 70B parameter models.\n"
            "3. Research GPU prices from at least 3 retailers.\n"
            "4. Build a detailed comparison table in CSV format with all data.\n"
            "5. Write analysis code in Python to calculate tok/s per dollar for each combo.\n"
            "6. Run the analysis code and interpret the results.\n"
            "7. Query a database to check if we have any prior benchmark data to compare against.\n"
            "8. Write a detailed executive summary with charts described in ASCII, specific "
            "recommendations for different budget tiers, and caveats.\n\n"
            "Show your complete step-by-step plan with tool calls for each step. "
            "Include hypothetical tool responses and your analysis of each."
        ),
    ),
    Scenario(
        name="instruction_following",
        description="Structured formatting — DRY relevance, constrained output",
        system=(
            "You are a precise technical assistant. Follow ALL formatting instructions "
            "exactly as specified. Do not deviate from the requested format."
        ),
        user=(
            "Answer each of the following questions using the EXACT format specified:\n\n"
            "Q1: What are the main differences between TCP and UDP?\n"
            "FORMAT: A markdown table with columns: Feature | TCP | UDP\n"
            "Include at least 8 rows covering: connection type, reliability, ordering, "
            "speed, header size, use cases, flow control, and error checking.\n\n"
            "Q2: Explain the CAP theorem.\n"
            "FORMAT: Three numbered sections (1. Consistency, 2. Availability, "
            "3. Partition Tolerance), each with exactly 3 bullet points. Then a paragraph "
            "explaining why you can only have two of three.\n\n"
            "Q3: List the SOLID principles.\n"
            "FORMAT: For each principle, provide:\n"
            "  - **Acronym letter**: Full name\n"
            "  - One-sentence definition\n"
            "  - A concrete Python code example (3-5 lines)\n"
            "  - A one-sentence anti-pattern\n\n"
            "Q4: Compare REST vs GraphQL vs gRPC.\n"
            "FORMAT: Three subsections with H3 headers. Each subsection must contain: "
            "a 2-sentence summary, a 'Pros' bullet list (3 items), a 'Cons' bullet list "
            "(3 items), and a 'Best for' one-liner.\n\n"
            "Q5: Describe the OAuth 2.0 authorization code flow.\n"
            "FORMAT: A numbered step-by-step list (at least 8 steps) where each step "
            "names the actor (Client, Auth Server, Resource Server, User) and the HTTP "
            "method used. End with a sequence diagram in ASCII art."
        ),
    ),
    Scenario(
        name="tool_calling",
        description="Tool call JSON generation — ngram excels at templates",
        system=(
            "You are a helpful travel planning assistant with access to tools. "
            "You MUST use tools for every factual query. For each tool call, first "
            "explain your reasoning in 2-3 sentences, then make the call. After all "
            "tool calls, write a detailed travel summary with packing recommendations."
        ),
        user=(
            "I'm planning a weekend trip visiting three cities: San Francisco, "
            "Seattle, and Portland. For each city:\n"
            "1. Check the current weather in Fahrenheit\n"
            "2. Convert each temperature to Celsius\n"
            "3. Search for outdoor events happening this weekend\n"
            "4. Search for food events happening this weekend\n\n"
            "After gathering all information, write a detailed day-by-day itinerary "
            "covering all three cities with weather-appropriate clothing suggestions "
            "for each day. Use the available tools for every lookup."
        ),
        tools=TOOLS_WEATHER,
    ),
]


# Convenience lookup
SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
