"""Sandboxed Python code execution tool for MCP server."""

import json
import logging
import multiprocessing
import os
import textwrap
from pathlib import Path
from typing import Annotated, Any

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field

from src.config import settings
from src.output_store import output_store
from src.tools._report import error_report

logger = logging.getLogger(__name__)

# Allowed imports for the sandbox
_ALLOWED_IMPORTS = {
    "numpy", "pandas", "matplotlib", "scipy", "sympy", "math", "statistics",
    "json", "csv", "re", "datetime", "collections", "itertools", "functools",
    "io", "os.path", "pathlib", "typing", "enum", "abc", "copy", "pprint",
    "hashlib", "base64", "struct", "urllib.parse", "urllib.request",
    "random", "string", "time", "unicodedata", "textwrap", "operator",
    "decimal", "fractions", "calendar", "tempfile", "shelve", "sqlite3",
}

# Blocked modules that must never be imported
_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "httpx", "aiohttp", "ftplib", "smtplib", "poplib",
    "imaplib", "telnetlib", "xmlrpc", "ctypes", "multiprocessing",
    "threading", "concurrent", "signal", "mmap", "select", "selectors",
    "asyncio", "ssl", "_thread", "posix", "nt", "winreg", "msvcrt",
    "termios", "tty", "pwd", "grp", "resource", "fcntl", "termios",
    "curses", "dbm", "gdbm", "nis", "sunau", "ossaudiodev",
    "audioop", "chunk", "colorsys", "crypt", "imghdr", "nntplib",
    "pipes", "sunau", "xdrlib", "aifc", "sndhdr", "wave",
}

# Timeout for code execution (seconds).
# NOTE: timeout is global — concurrent calls share this setting.
_CODE_EXEC_TIMEOUT = 30


def _check_import_allowed(module_name: str) -> bool:
    """Check if a module import is allowed in the sandbox."""
    # Block top-level blocked modules
    top_level = module_name.split(".")[0]
    if top_level in _BLOCKED_IMPORTS:
        return False
    # Allow if in allowed list or if it's a submodule of an allowed module
    if module_name in _ALLOWED_IMPORTS:
        return True
    # Check if top-level is in allowed imports
    if top_level in _ALLOWED_IMPORTS:
        return True
    # Allow stdlib modules not in blocked list
    try:
        from importlib import util as _util
        spec = _util.find_spec(module_name)
        if spec and spec.origin:
            # If it's in the Python stdlib (not in site-packages), allow it
            origin = spec.origin
            if "site-packages" not in origin and "dist-packages" not in origin:
                return True
    except Exception:
        pass
    return False


def _wrap_code(code: str) -> str:
    """Wrap user code with safety hooks and output capture."""
    # Create a restricted import hook
    wrapper = textwrap.dedent("""
        import builtins
        import sys
        import io
        import traceback

        # Override __import__ to block dangerous imports
        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if not _check_import_allowed(name):
                raise ImportError(f"Import of '{{name}}' is not allowed in sandbox.")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = restricted_import

        # Capture stdout and stderr
        _stdout = io.StringIO()
        _stderr = io.StringIO()
        sys.stdout = _stdout
        sys.stderr = _stderr

        # Import guard function for sandbox
        _allowed_imports = {allowed_imports}
        _blocked_imports = {blocked_imports}

        def _check_import_allowed(module_name):
            top_level = module_name.split(".")[0]
            if top_level in _blocked_imports:
                return False
            if module_name in _allowed_imports:
                return True
            if top_level in _allowed_imports:
                return True
            try:
                from importlib import util as _util
                spec = _util.find_spec(module_name)
                if spec and spec.origin:
                    origin = spec.origin
                    if "site-packages" not in origin and "dist-packages" not in origin:
                        return True
            except Exception:
                pass
            return False

        try:
            exec_result = None
            try:
                exec_result = eval(compile({code_literal}, "<sandbox>", "eval"))
            except SyntaxError:
                exec(compile({code_literal}, "<sandbox>", "exec"))
        except Exception as e:
            _stderr.write(traceback.format_exc())
        finally:
            stdout_text = _stdout.getvalue()
            stderr_text = _stderr.getvalue()

            # Write results to output file
            with open({output_file}, "w") as f:
                import json as _json
                result = {{
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "result": str(exec_result) if exec_result is not None else None,
                }}
                _json.dump(result, f, indent=2, ensure_ascii=False)
    """)

    code_escaped = json.dumps(code)

    return wrapper.format(
        allowed_imports=repr(_ALLOWED_IMPORTS),
        blocked_imports=repr(_BLOCKED_IMPORTS),
        code_literal=code_escaped,
        output_file=json.dumps("/tmp/code_run_output.json"),
    )


def _exec_wrapper(code: str) -> None:
    """Execute wrapped code in subprocess (must be top-level for multiprocessing)."""
    # Use same dict for globals+locals so _check_import_allowed is in global scope
    # and accessible to restricted_import called via builtins.__import__
    ns = {}
    exec(compile(code, "<sandbox>", "exec"), ns, ns)  # type: ignore[arg-type]


def _run_code_sandbox(code: str, timeout: int) -> dict[str, Any]:
    """Run code in a subprocess sandbox with a per-call timeout (seconds)."""
    wrapped = _wrap_code(code)
    output_file = "/tmp/code_run_output.json"
    # Remove stale output
    if os.path.exists(output_file):
        os.remove(output_file)

    proc = multiprocessing.Process(
        target=_exec_wrapper,
        args=(wrapped,),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "status": "error",
            "error": f"Code execution timed out after {timeout}s",
            "exit_code": -1,
        }

    if proc.exitcode != 0:
        return {
            "status": "error",
            "error": f"Process exited with code {proc.exitcode}",
            "exit_code": proc.exitcode,
        }

    if not os.path.exists(output_file):
        return {
            "status": "error",
            "error": "Code execution failed — no output produced",
            "exit_code": 1,
        }

    try:
        with open(output_file, "r") as f:
            result = json.load(f)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to read execution output: {str(e)}",
            "exit_code": 1,
        }

    # Return full stdout/stderr; the handler paginates oversized output via read_output.
    return {
        "status": "success",
        "exit_code": 0,
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "result": result.get("result"),
    }


def _code_error_hint(message: str, exit_code: int | None) -> str | None:
    """One-line corrective hint for a sandbox-level failure."""
    msg = (message or "").lower()
    if "timed out" in msg:
        return (
            "Computation exceeded the time limit. Pass a larger timeout=<seconds>, "
            "or reduce the work (e.g. vectorize with numpy, process less data)."
        )
    # SIGKILL (-9) almost always means the OOM killer reclaimed the process.
    if exit_code in (-9, 137) or "code -9" in msg:
        return "Process was killed — most likely out of memory. Reduce data size or memory use."
    return None


def _stderr_hint(stderr: str) -> str | None:
    """Hint when stderr reveals a blocked/failed import (runs but errors at runtime)."""
    s = (stderr or "").lower()
    if "not allowed in sandbox" in s or "importerror" in s or "modulenotfounderror" in s:
        return (
            "An import failed or was blocked. Allowed: numpy, pandas, matplotlib, scipy, "
            "sympy, math, statistics, json, csv, re, datetime, collections, itertools, "
            "functools, io, os.path, pathlib (+ most stdlib). Network/OS modules (os, sys, "
            "subprocess, socket, http, urllib, requests) are blocked by design — for web "
            "access use fetch/search/navigate_page/click/fill/evaluate instead."
        )
    return None


def code_run_handler(server: FastMCP) -> None:
    """Register the code_run tool."""

    @server.tool()
    async def code_run(
        code: Annotated[str, Field(description="Python source to execute. Use print() to emit output; the last expression is also captured.")],
        timeout: Annotated[
            int | None,
            Field(description="Max seconds before the sandbox is killed (default 30). Raise for heavy computation."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Execute Python code in a sandboxed subprocess. timeout defaults to 30s.

        Allowed: numpy, pandas, matplotlib, scipy, sympy, math, statistics, json, csv,
        re, datetime, collections, itertools, functools, io, os.path, pathlib.
        Blocked: os, sys, subprocess, shutil, socket, http, urllib, requests,
        threading, multiprocessing, signal, mmap. (For web access use fetch/search/
        navigate_page, click, fill, evaluate, not code_run.)
        Use print() to return values; last expression result is also captured.
        Large stdout/stderr is previewed with a read_output handle shown in its
        section; call read_output(handle=...) to read the rest.
        Returns: markdown — status line, stdout/stderr as code blocks (omitted when
        empty), and the final expression result. On failure: an Error block plus a
        hint (e.g. raise timeout=, OOM, or a blocked import).
        """
        eff_timeout = timeout or _CODE_EXEC_TIMEOUT

        try:
            if ctx:
                await ctx.report_progress(0, 1, "Running code in sandbox\u2026")
            result = _run_code_sandbox(code, eff_timeout)
            if ctx:
                await ctx.report_progress(1, 1, "Done")
            if result.get("status") != "success":
                msg = result.get("error", "unknown error")
                exit_code = result.get("exit_code")
                if exit_code is not None:
                    msg = f"{msg} (exit code {exit_code})"
                return error_report(msg, _code_error_hint(msg, exit_code))

            holder: dict[str, Any] = {}
            output_store.attach(holder, "stdout", result.get("stdout", "") or "", source="code_run stdout")
            output_store.attach(holder, "stderr", result.get("stderr", "") or "", source="code_run stderr")

            parts = [f"**Status:** success · exit {result.get('exit_code', 0)}"]

            stdout = holder.get("stdout", "")
            if stdout:
                parts += ["", "**stdout:**", "```text", stdout, "```"]
                if "stdout_hint" in holder:
                    parts.append(f"_{holder['stdout_hint']}_")

            stderr = holder.get("stderr", "")
            if stderr:
                parts += ["", "**stderr:**", "```text", stderr, "```"]
                if "stderr_hint" in holder:
                    parts.append(f"_{holder['stderr_hint']}_")
                # The process exited 0 but raised at runtime (traceback in stderr).
                runtime_hint = _stderr_hint(result.get("stderr", ""))
                if runtime_hint:
                    parts.append(f"_{runtime_hint}_")

            res = result.get("result")
            if res is not None:
                parts += ["", f"**result:** `{res}`"]

            return "\n".join(parts)
        except Exception as e:
            logger.error("Code run error: %s", str(e))
            return error_report(str(e))

    logger.info("Registered code_run tool")
