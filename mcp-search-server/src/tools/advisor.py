"""Advisor tool — calls the local LLM for expert reasoning on complex problems."""

import asyncio
import logging
from typing import Annotated, Any

import httpx
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field

from src.config import settings
from src.output.format import format_result, format_error
from src.tools._report import error_report as _legacy_error_report

logger = logging.getLogger(__name__)


def _build_messages(context: str, question: str) -> list[dict[str, Any]]:
    """Build the chat messages for the advisor prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert advisor. Reason carefully through the context and "
                "question, then give precise, actionable guidance. If context is "
                "insufficient, state exactly what is missing."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]


def _format_response(response_text: str) -> str:
    """Format the LLM response for MCP consumption."""
    return response_text.strip()


async def _wait_for_model_ready(
    base_url: str,
    api_key: str,
    on_status: Any = None,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
) -> None:
    """Poll the llama.cpp /health endpoint until the model is ready.

    Args:
        base_url: OpenAI-compatible base URL (e.g. http://llama-server:8080/v1).
                  The /v1 suffix is stripped to reach the server root.
        api_key: Optional bearer token.
        on_status: Async callable(message: str) invoked on each non-ok poll.
        poll_interval: Seconds between polls.
        timeout: Max seconds to wait before giving up (proceeds anyway).
    """
    # Derive server root from base_url (strip trailing /v1 or /v1/)
    server_root = base_url.rstrip("/")
    if server_root.endswith("/v1"):
        server_root = server_root[:-3]
    health_url = f"{server_root}/health"

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    deadline = asyncio.get_event_loop().time() + timeout
    dots = 0
    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(health_url, headers=headers)
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                status = data.get("status", "")
                if resp.status_code == 200 and status == "ok":
                    return
                dots = (dots % 3) + 1
                msg = f"Model loading{'.' * dots}"
                if status:
                    msg = f"{status}{'.' * dots}"
            except Exception:
                dots = (dots % 3) + 1
                msg = f"Waiting for server{'.' * dots}"
            if on_status is not None:
                await on_status(msg)
            await asyncio.sleep(poll_interval)
    # Timed out — proceed and let the inference call fail if still not ready


async def _call_llm_stream(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    on_chunk: Any = None,
) -> str:
    """Call the local LLM via OpenAI-compatible streaming API.

    Yields text chunks via ``on_chunk(text)`` as they arrive and returns the
    complete response when done.  Falls back to a regular (non-streaming) call
    if the server does not support SSE.
    """
    import json as _json

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": settings.ADVISOR_MAX_TOKENS,
        "stream": True,
    }

    accumulated: list[str] = []

    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content") or ""
                if text:
                    accumulated.append(text)
                    if on_chunk is not None:
                        await on_chunk(text)

    return "".join(accumulated)


async def _call_llm(messages: list[dict], model: str, base_url: str, api_key: str) -> str:
    """Call the local LLM via OpenAI-compatible API (non-streaming fallback)."""
    return await _call_llm_stream(messages, model, base_url, api_key)


async def call_advisor(
    context: str,
    question: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call the local LLM as an advisor.

    Args:
        context: The problem context or background information.
        question: The specific question to ask the advisor.
        model: The model to use (defaults to settings.ADVISOR_MODEL).
        base_url: The LLM server URL (defaults to settings.ADVISOR_BASE_URL).
        api_key: Optional API key for the LLM server (defaults to settings.ADVISOR_API_KEY).

    Returns:
        The advisor's response.
    """
    # Use config defaults as fallbacks
    effective_model = model or settings.ADVISOR_MODEL
    effective_base_url = base_url or settings.ADVISOR_BASE_URL
    effective_api_key = api_key or settings.ADVISOR_API_KEY

    messages = _build_messages(context, question)
    response = await _call_llm(messages, effective_model, effective_base_url, effective_api_key)
    return _format_response(response)


def advisor_handler(server: FastMCP) -> None:
    """Register the advisor tool."""

    @server.tool()
    async def advisor(
        context: Annotated[
            str,
            Field(description="All relevant background for the question — be generous; the advisor only sees what you pass."),
        ],
        question: Annotated[str, Field(description="The specific question or task to reason about.")],
        ctx: Context | None = None,
    ) -> str:
        """Ask the advisor LLM for expert reasoning. Use early and often.

        Calls a strong local model for deeper analysis than your own. Reach for it
        when stuck, planning a multi-step task, or unsure of an approach.

        context: all relevant background (be generous).
        question: the specific question or task.
        Returns: compact text — [advisor: model] header followed by the response.
        """
        try:
            accumulated: list[str] = []

            async def _on_status(msg: str) -> None:
                if ctx:
                    await ctx.report_progress(0, None, msg)

            async def _on_chunk(text: str) -> None:
                accumulated.append(text)
                if ctx:
                    await ctx.report_progress(
                        len(accumulated),
                        None,
                        "".join(accumulated),
                    )

            if ctx:
                await ctx.report_progress(0, None, f"Consulting advisor ({settings.ADVISOR_MODEL})\u2026")

            effective_model = settings.ADVISOR_MODEL
            effective_base_url = settings.ADVISOR_BASE_URL
            effective_api_key = settings.ADVISOR_API_KEY

            await _wait_for_model_ready(effective_base_url, effective_api_key, on_status=_on_status)

            messages = _build_messages(context, question)
            result = await _call_llm_stream(
                messages, effective_model, effective_base_url, effective_api_key, on_chunk=_on_chunk
            )
            result = _format_response(result)

            if ctx:
                await ctx.report_progress(len(accumulated), len(accumulated), result)
            return f"[advisor: {settings.ADVISOR_MODEL}]\n\n{result}"
        except Exception as e:
            logger.error("Advisor error: %s", str(e))
            hint = (
                "The advisor LLM is unreachable or errored. Proceed without it using "
                "your own reasoning and the other tools, or retry once."
            )
            return format_error(f"Advisor ({settings.ADVISOR_MODEL}): {e}", hint=hint)

    logger.info("Registered advisor tool")
