"""Shared helpers for rendering consistent, recoverable tool error reports.

Tools return rich, actionable errors (status line + an Error block + a
one-line corrective hint) so the model can self-correct. These helpers let all
tools (code_run, read_output) return the same shape instead of a
bare exception string the model can't act on.

Backward-compatible: the legacy error_report format is preserved for tests
that assert on the old markdown shape. New tools should use
output.format.format_error for compact output.
"""

from src.output.format import format_error as _format_error
from src.output.format import FOOTER_SEP


def error_report(message: str, hint: str | None = None, *, status: str = "error") -> str:
    """Render a uniform markdown error report (legacy format for backward compat).

    Args:
        message: The error text (traceback, exception message, etc.).
        hint: Optional one-line corrective guidance shown in italics.
        status: Status label for the first line (default "error").

    Returns:
        Markdown: a ``**Status:**`` line, an ``**Error:**`` code block, and the
        optional hint.
    """
    body = (message or "").strip() or "(no message)"
    parts = [f"**Status:** {status}", "", "**Error:**", "```text", body, "```"]
    if hint:
        parts.append(f"_{hint}_")
    return "\n".join(parts)
