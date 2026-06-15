"""Advisor tool — calls the local LLM for expert reasoning on complex problems."""

import json
import logging
from typing import Any

import httpx
from mcp.server import FastMCP

from src.config import settings

logger = logging.getLogger(__name__)


def _build_messages(context: str, question: str) -> list[dict[str, Any]]:
    """Build the chat messages for the advisor prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert advisor with deep reasoning capabilities. "
                "Your role is to analyze the user's context and question carefully, "
                "then provide thorough, well-reasoned guidance. "
                "Be precise, thorough, and helpful. If the context is insufficient, "
                "state what additional information would help."
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
        context: str,
        question: str,
        model: str | None = None,
    ) -> str:
        """Ask the advisor model for expert guidance on a problem.

        This tool calls your local LLM server (qwopus3.6-27b by default) to get
        expert reasoning on complex problems. Use it when you need deeper analysis
        than your primary model can provide.

        Args:
            context: The problem context or background information (as much detail as needed).
            question: The specific question or task to ask the advisor.
            model: The model to use (overrides config default: qwopus3.6-27b).

        Returns:
            The advisor's response with analysis and guidance.
        """
        try:
            result = await call_advisor(context, question, model)
            return json.dumps({
                "status": "success",
                "model": model or settings.ADVISOR_MODEL,
                "response": result,
            }, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Advisor error: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
            }, indent=2)

    logger.info("Registered advisor tool")
