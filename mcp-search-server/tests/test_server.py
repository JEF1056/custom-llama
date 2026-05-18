"""Tests for the MCP Search Server."""

import json
import os
import tempfile
from pathlib import Path

import pytest


def test_import():
    """Test that the server can be imported."""
    from src.server import create_server

    server = create_server()
    assert server is not None


def test_config():
    """Test that the config can be imported."""
    from src.config import settings

    assert settings.MCP_SERVER_HOST == "0.0.0.0"
    assert settings.MCP_SERVER_PORT == 3100
    assert settings.SEARCH_ENGINE == "duckduckgo"
    assert settings.MAX_RESULTS == 10


def test_search_models():
    """Test the search models."""
    from src.search.models import SearchResult, SearchResponse

    result = SearchResult(
        title="Test",
        url="https://example.com",
        snippet="Test snippet",
        engine="duckduckgo",
    )
    assert result.title == "Test"
    assert result.url == "https://example.com"
    assert result.snippet == "Test snippet"

    response = SearchResponse(query="test", results=[result], total=1)
    assert response.query == "test"
    assert len(response.results) == 1
    assert response.total == 1


def test_http_request_tool_registered():
    """Test that http_request tool is registered."""
    from src.server import create_server, register_tools

    server = create_server()
    register_tools(server)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "http_request" in tool_names


def test_http_request_basic():
    """Test http_request with a simple GET."""
    import asyncio
    from src.tools.http_request import http_request_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    http_request_handler(server)
    fn = server.tools["http_request"]

    result_str = asyncio.run(fn("GET", "https://httpbin.org/get"))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert result["status_code"] == 200
    assert "elapsed_seconds" in result


def test_http_request_auth_bearer():
    """Test http_request with bearer auth."""
    import asyncio
    from src.tools.http_request import http_request_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    http_request_handler(server)
    fn = server.tools["http_request"]

    result_str = asyncio.run(fn(
        "GET",
        "https://httpbin.org/headers",
        auth={"type": "bearer", "value": "test-token"},
    ))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert result["status_code"] == 200


def test_http_request_unsupported_method():
    """Test http_request rejects unsupported methods."""
    import asyncio
    from src.tools.http_request import http_request_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    http_request_handler(server)
    fn = server.tools["http_request"]

    result_str = asyncio.run(fn("TRACE", "https://example.com"))
    result = json.loads(result_str)
    assert result["status"] == "error"


def test_time_now_tool_registered():
    """Test that time_now tool is registered."""
    from src.server import create_server, register_tools

    server = create_server()
    register_tools(server)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "time_now" in tool_names


def test_time_now_default():
    """Test time_now returns current time."""
    import asyncio
    from src.tools.time_now import time_now_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    time_now_handler(server)
    fn = server.tools["time_now"]

    result_str = asyncio.run(fn())
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert "iso_format" in result
    assert "date" in result
    assert "time" in result
    assert "day_of_week" in result


def test_time_now_timezone():
    """Test time_now with specific timezone."""
    import asyncio
    from src.tools.time_now import time_now_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    time_now_handler(server)
    fn = server.tools["time_now"]

    result_str = asyncio.run(fn(timezone_name="Asia/Tokyo"))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert result["timezone"] == "Asia/Tokyo"


def test_time_now_conversion():
    """Test time_now timezone conversion."""
    import asyncio
    from src.tools.time_now import time_now_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    time_now_handler(server)
    fn = server.tools["time_now"]

    result_str = asyncio.run(fn(
        timezone_name="America/New_York",
        convert_from_timezone="Asia/Tokyo",
        convert_from_time="2026-01-15 14:30:00",
    ))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert result["operation"] == "conversion"
    assert "converted_time" in result


def test_calculator_tool_registered():
    """Test that calculator tool is registered."""
    from src.server import create_server, register_tools

    server = create_server()
    register_tools(server)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "calculator" in tool_names


def test_calculator_basic():
    """Test calculator basic arithmetic."""
    import asyncio
    from src.tools.calculator import calculator_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    calculator_handler(server)
    fn = server.tools["calculator"]

    result_str = asyncio.run(fn("2 + 3 * 4"))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert result["result"] == "14"


def test_calculator_symbolic():
    """Test calculator symbolic math."""
    import asyncio
    from src.tools.calculator import calculator_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    calculator_handler(server)
    fn = server.tools["calculator"]

    result_str = asyncio.run(fn("solve(x**2 - 4, x)"))
    result = json.loads(result_str)
    assert result["status"] == "success"


def test_calculator_matrix():
    """Test calculator matrix operations."""
    import asyncio
    from src.tools.calculator import calculator_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    calculator_handler(server)
    fn = server.tools["calculator"]

    result_str = asyncio.run(fn("Matrix([[1,2],[3,4]]).det()"))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert result["result"] == "-2"


def test_file_ops_tool_registered():
    """Test that file operations tools are registered."""
    from src.server import create_server, register_tools

    server = create_server()
    register_tools(server)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "file_read" in tool_names
    assert "file_list" in tool_names
    assert "file_delete" in tool_names


def test_file_ops_create_and_read():
    """Test file create, read, and delete cycle."""
    import asyncio
    from src.tools.file_ops import file_operations_handler
    from src.tools.filetool import create_file_handler
    from src.config import settings

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    create_file_handler(server)
    file_operations_handler(server)

    # Create a file
    create_fn = server.tools["create_file"]
    read_fn = server.tools["file_read"]
    list_fn = server.tools["file_list"]
    delete_fn = server.tools["file_delete"]

    # Create
    result = asyncio.run(create_fn(
        filename="test.txt",
        content="Hello, world!",
    ))
    assert len(result) == 2  # embedded resource + text

    # Read
    result_str = asyncio.run(read_fn("test.txt"))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert result["content"] == "Hello, world!"

    # List
    result_str = asyncio.run(list_fn())
    result = json.loads(result_str)
    assert result["status"] == "success"
    filenames = [e["name"] for e in result["entries"]]
    assert "test.txt" in filenames

    # Delete
    result_str = asyncio.run(delete_fn("test.txt"))
    result = json.loads(result_str)
    assert result["status"] == "success"


def test_file_ops_path_traversal_blocked():
    """Test that path traversal is blocked in file ops."""
    import asyncio
    from src.tools.file_ops import file_operations_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    file_operations_handler(server)
    read_fn = server.tools["file_read"]

    result_str = asyncio.run(read_fn("../etc/passwd"))
    result = json.loads(result_str)
    assert result["status"] == "error"


def test_code_run_tool_registered():
    """Test that code_run tool is registered."""
    from src.server import create_server, register_tools

    server = create_server()
    register_tools(server)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "code_run" in tool_names


def test_code_run_basic():
    """Test code_run with simple code."""
    import asyncio
    from src.tools.code_run import code_run_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    code_run_handler(server)
    fn = server.tools["code_run"]

    result_str = asyncio.run(fn('print("hello")'))
    result = json.loads(result_str)
    assert result["status"] == "success"
    assert "hello" in result["stdout"]


def test_code_run_blocked_import():
    """Test that dangerous imports are blocked."""
    import asyncio
    from src.tools.code_run import code_run_handler

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    server = FakeServer()
    code_run_handler(server)
    fn = server.tools["code_run"]

    result_str = asyncio.run(fn('import os'))
    result = json.loads(result_str)
    # Should error because os is blocked
    assert result["status"] == "error" or "ImportError" in result.get("stderr", "")


def test_content_extractor():
    """Test content extractor basic functionality."""
    from src.extractor.content import ContentExtractor

    extractor = ContentExtractor()
    html = """<html><head><title>Test</title></head>
    <body><p>Hello world</p><script>alert('xss')</script></body></html>"""
    result = extractor.extract(html)
    assert result["title"] == "Test"
    assert "Hello world" in result["content"]
    assert "alert" not in result["content"]


def test_content_extractor_truncation():
    """Test content extractor truncation."""
    from src.extractor.content import ContentExtractor

    extractor = ContentExtractor()
    html = "<html><body><p>" + "x" * 5000 + "</p></body></html>"
    result = extractor.extract(html, max_length=100, truncate=True)
    assert len(result["content"]) <= 100
    assert "Summarized" in result["content"]


def test_all_tools_registered():
    """Test that all expected tools are registered."""
    from src.server import create_server, register_tools

    server = create_server()
    register_tools(server)
    tool_names = [t.name for t in server._tool_manager.list_tools()]

    expected = {
        "search", "fetch", "deep_search",
        "browser_create_session", "browser_navigate", "browser_screenshot",
        "browser_click", "browser_fill", "browser_evaluate",
        "browser_get_text", "browser_get_content", "browser_monitor",
        "browser_close", "browser_list_sessions",
        "create_file", "file_read", "file_list", "file_delete",
        "http_request", "code_run", "time_now", "calculator",
    }
    for name in expected:
        assert name in tool_names, f"Tool {name} not registered"
