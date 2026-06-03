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
        system="You are a thoughtful essayist who writes in clear, engaging prose.",
        user=(
            "Write a detailed essay analyzing how artificial intelligence is transforming "
            "modern education. Structure your essay with an introduction, three body sections "
            "(benefits, risks, and policy recommendations), and a conclusion. Use specific "
            "examples and cite hypothetical studies to support each point. Each section should "
            "be at least two paragraphs. Do not use bullet points or numbered lists — write "
            "in flowing paragraph form only."
        ),
    ),
    Scenario(
        name="coding",
        description="Repetitive syntax — ngram speculation advantage",
        system="You are an expert Python developer. Write clean, production-ready code with comprehensive type hints and docstrings.",
        user=(
            "Write a Python module that implements a `TaskScheduler` class with the following "
            "features:\n"
            "1. A priority queue backed by `heapq` where tasks have a name, priority (int), "
            "and an async callable.\n"
            "2. Delayed execution: tasks can be scheduled to run after a specified delay in "
            "seconds.\n"
            "3. Retry logic: if a task raises an exception, retry up to N times with "
            "exponential backoff (base 2 seconds).\n"
            "4. A worker pool of configurable size that pulls tasks from the queue "
            "concurrently using `asyncio`.\n"
            "5. Graceful shutdown: cancel pending tasks and wait for running tasks to finish.\n"
            "6. Include a `TaskResult` dataclass that captures task name, success/failure, "
            "duration, retry count, and error message if any.\n\n"
            "Write the complete module with all imports, classes, and a brief `if __name__` "
            "demo that schedules 5 example tasks."
        ),
    ),
    Scenario(
        name="agentic",
        description="Multi-step reasoning — tests mixed content decode",
        system=(
            "You are an AI research agent. You have access to the following tools:\n"
            "- web_search(query: str) → list of search results with titles and snippets\n"
            "- file_read(path: str) → file contents as string\n"
            "- code_execute(language: str, code: str) → execution output\n"
            "- database_query(sql: str) → query results as JSON\n\n"
            "For each step, show your reasoning, which tool you would call and why, "
            "and what you expect the result to look like. Be thorough and methodical."
        ),
        user=(
            "I need you to research and compile a comprehensive performance comparison of "
            "LLM inference engines on consumer GPUs (RTX 3090, 4090, 4080). Specifically:\n"
            "1. Find the latest benchmarks for vLLM, llama.cpp, TensorRT-LLM, and SGLang.\n"
            "2. Compare tokens/second for 7B, 13B, and 27B parameter models.\n"
            "3. Build a comparison table in CSV format.\n"
            "4. Calculate which engine gives the best tok/s per dollar of GPU cost.\n"
            "5. Write an executive summary with a recommendation.\n\n"
            "Show your complete step-by-step plan with tool calls for each step."
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
        system="You are a helpful assistant with access to tools.",
        user=(
            "I'm planning a weekend trip to San Francisco. Can you:\n"
            "1. Check the current weather in San Francisco in Fahrenheit\n"
            "2. Convert that temperature to Celsius so I know what to pack\n"
            "3. Search for outdoor events happening this weekend in San Francisco\n\n"
            "Use the available tools to help me with each of these tasks."
        ),
        tools=TOOLS_WEATHER,
    ),
]


# Convenience lookup
SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
