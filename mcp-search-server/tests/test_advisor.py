"""Tests for the advisor tool — calls the local LLM for expert reasoning."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock browser module before any src.tools imports — browser.py has
# side effects (BrowserManager init) that fail in non-container envs.
# We mock src.browser.automation which is the actual module imported by browser.py.
_mock_automation = MagicMock()
_mock_automation.get_browser_manager.return_value = MagicMock()
sys.modules.setdefault("src.browser.automation", _mock_automation)
sys.modules.setdefault("src.browser", type(sys)("src.browser"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tools.advisor import (
    _build_messages,
    _call_llm,
    _format_response,
    advisor_handler,
    call_advisor,
)
from config import settings
from server import create_server, register_tools


class FakeServer:
    """Mock MCP server for testing tool registration."""

    def __init__(self):
        self.tools = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


def _run(coro, **kwargs):
    """Run an async coroutine synchronously for testing."""
    return asyncio.run(coro)


# --- Tool registration ---


def test_advisor_tool_registered():
    """Test that advisor tool is registered via the real server."""
    server = create_server()
    register_tools(server)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "advisor" in tool_names


def test_advisor_handler_registers_tool():
    """Test that advisor_handler registers the advisor tool."""
    server = FakeServer()
    advisor_handler(server)
    assert "advisor" in server.tools


# --- _build_messages ---


def test_build_messages_structure():
    """Test that _build_messages produces correct message structure."""
    messages = _build_messages("Some context", "What should I do?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "expert advisor" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"
    assert "Some context" in messages[1]["content"]
    assert "What should I do?" in messages[1]["content"]


def test_build_messages_escapes_special_chars():
    """Test that context with special characters is preserved."""
    context = "Use <html> & symbols in context"
    question = "How do I handle this?"
    messages = _build_messages(context, question)
    assert "<html>" in messages[1]["content"]
    assert "&" in messages[1]["content"]


# --- _format_response ---


def test_format_response_strips_whitespace():
    """Test that _format_response strips leading/trailing whitespace."""
    assert _format_response("  hello  ") == "hello"
    assert _format_response("\n\nhello\n\n") == "hello"
    assert _format_response("hello") == "hello"


# --- _call_llm ---


@pytest.mark.asyncio
async def test_call_llm_success():
    """Test that _call_llm correctly calls the LLM endpoint."""
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is the advisor's response.",
                    "role": "assistant",
                }
            }
        ]
    }

    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        result = await _call_llm(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            base_url="http://test.example.com/v1",
            api_key="test-key",
        )

    assert result == "This is the advisor's response."
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[1]["json"]["model"] == "test-model"
    assert call_kwargs[1]["json"]["max_tokens"] == settings.ADVISOR_MAX_TOKENS


@pytest.mark.asyncio
async def test_call_llm_no_choices():
    """Test that _call_llm raises ValueError when no choices returned."""
    mock_response = {"choices": []}
    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="No choices returned"):
            await _call_llm(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
                base_url="http://test.example.com/v1",
                api_key="",
            )


@pytest.mark.asyncio
async def test_call_llm_http_error():
    """Test that _call_llm propagates HTTP errors."""
    import httpx

    mock_response_obj = AsyncMock()
    mock_response_obj.raise_for_status = MagicMock(side_effect=httpx.HTTPError("Connection failed"))
    # json() must return a dict (not a coroutine) since we await it
    mock_response_obj.json = AsyncMock(return_value={"choices": []})

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.HTTPError):
            await _call_llm(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
                base_url="http://test.example.com/v1",
                api_key="",
            )


# --- call_advisor ---


@pytest.mark.asyncio
async def test_call_advisor_uses_config_defaults():
    """Test that call_advisor uses config defaults when no overrides provided."""
    mock_response = {
        "choices": [{"message": {"content": "Advisor says hello"}}]
    }
    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        result = await call_advisor(
            context="test context",
            question="test question",
        )

    assert result == "Advisor says hello"
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["model"] == settings.ADVISOR_MODEL
    assert payload["max_tokens"] == settings.ADVISOR_MAX_TOKENS


@pytest.mark.asyncio
async def test_call_advisor_uses_override_model():
    """Test that call_advisor uses override model when provided."""
    mock_response = {
        "choices": [{"message": {"content": "Advisor says hello"}}]
    }
    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        result = await call_advisor(
            context="test context",
            question="test question",
            model="custom-model",
        )

    assert result == "Advisor says hello"
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["model"] == "custom-model"


# --- advisor_handler ---


def test_advisor_handler_success():
    """Test that advisor_handler returns success JSON on successful call."""
    mock_response = {
        "choices": [{"message": {"content": "Advisor response"}}]
    }
    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        server = FakeServer()
        advisor_handler(server)
        advisor_fn = server.tools["advisor"]

        async def _test():
            result = await advisor_fn(
                context="test context",
                question="test question",
                model="test-model",
            )
            return result

        result = _run(_test())
    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert parsed["model"] == "test-model"
    assert parsed["response"] == "Advisor response"


def test_advisor_handler_error():
    """Test that advisor_handler returns error JSON on failure."""
    import httpx

    mock_response_obj = AsyncMock()
    mock_response_obj.raise_for_status = MagicMock(side_effect=httpx.HTTPError("Connection failed"))
    mock_response_obj.json = AsyncMock(return_value={"choices": []})

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        server = FakeServer()
        advisor_handler(server)
        advisor_fn = server.tools["advisor"]

        async def _test():
            result = await advisor_fn(
                context="test context",
                question="test question",
            )
            return result

        result = _run(_test())
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert "Connection failed" in parsed["error"]


def test_advisor_handler_uses_config_default_model():
    """Test that advisor_handler uses config default when no model arg."""
    mock_response = {"choices": [{"message": {"content": "OK"}}]}
    mock_response_obj = AsyncMock()
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    mock_response_obj.raise_for_status = AsyncMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.advisor.httpx.AsyncClient", return_value=mock_client):
        server = FakeServer()
        advisor_handler(server)
        advisor_fn = server.tools["advisor"]

        async def _test():
            result = await advisor_fn(
                context="test context",
                question="test question",
            )
            return result

        result = _run(_test())
    parsed = json.loads(result)
    assert parsed["model"] == settings.ADVISOR_MODEL


# --- Config integration ---


def test_config_has_advisor_settings():
    """Test that config has all advisor-related settings."""
    assert hasattr(settings, "ADVISOR_BASE_URL")
    assert hasattr(settings, "ADVISOR_MODEL")
    assert hasattr(settings, "ADVISOR_API_KEY")
    assert hasattr(settings, "ADVISOR_MAX_TOKENS")


def test_config_defaults():
    """Test that config defaults match expected values."""
    assert settings.ADVISOR_BASE_URL == "http://localhost:8080/v1"
    assert settings.ADVISOR_MODEL == "qwopus3.6-27b"
    assert settings.ADVISOR_API_KEY == ""
    assert settings.ADVISOR_MAX_TOKENS == 8192
