"""Advisor tool — calls the local LLM for expert reasoning on complex problems."""

import logging
from typing import Annotated, Any

import httpx
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field

from src.config import settings
from src.tools._report import error_report

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


async def _call_llm(messages: list[dict], model: str, base_url: str, api_key: str) -> str:
    """Call the local LLM via OpenAI-compatible API."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": settings.ADVISOR_MAX_TOKENS,
    }

    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = await response.json()

    # Extract the assistant's message content
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices returned from LLM")

    message = choices[0].get("message", {})
    return message.get("content", "")


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
        Returns: markdown — the advisor's response under a header naming the model.
        """
        try:
            if ctx:
                await ctx.report_progress(0, 1, f"Consulting advisor ({settings.ADVISOR_MODEL})\u2026")
            result = await call_advisor(context, question)
            if ctx:
                await ctx.report_progress(1, 1, "Done")
            return f"**Advisor ({settings.ADVISOR_MODEL}):**\n\n{result}"
        except Exception as e:
            logger.error("Advisor error: %s", str(e))
            hint = (
                "The advisor LLM is unreachable or errored. Proceed without it using "
                "your own reasoning and the other tools, or retry once."
            )
            return error_report(f"Advisor ({settings.ADVISOR_MODEL}): {e}", hint)

    logger.info("Registered advisor tool")
