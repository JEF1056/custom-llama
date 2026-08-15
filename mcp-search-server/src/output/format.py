"""Output formatting helpers for token-efficient tool results.

All MCP tools use these conventions for consistent, compact output:
- FOOTER_SEP ("---") separates result content from metadata footers
- format_result() wraps text with optional footer
- format_error() renders uniform error reports
"""

FOOTER_SEP = "---"


def format_result(text: str, *, footer: str | None = None) -> str:
    """Format a tool result with optional footer separated by FOOTER_SEP.

    Args:
        text: The main result text.
        footer: Optional metadata footer (e.g. pagination hints).

    Returns:
        Formatted string with separator between text and footer.
    """
    if footer:
        return f"{text}\n{FOOTER_SEP}\n{footer}"
    return text


def format_error(message: str, *, hint: str | None = None) -> str:
    """Format a uniform error report.

    Args:
        message: The error text.
        hint: Optional one-line corrective guidance.

    Returns:
        Error message with optional hint separated by FOOTER_SEP.
    """
    body = (message or "").strip() or "(no message)"
    parts = [body]
    if hint:
        parts.append(FOOTER_SEP)
        parts.append(hint)
    return "\n".join(parts)
