"""Low-level harness: edit models.ini, drive the container, measure throughput.

All ini edits are *section-aware* — a key is replaced only inside its declared
section ([*] or [qwopus3.6-27b]) so e.g. ``ctx-size`` in the 35B block is never
touched. Editing the host file is sufficient: the read-only bind mount only
blocks container-side writes; host edits are re-read on force-recreate.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config as C


# ── api key ───────────────────────────────────────────────────────────────────
def _api_key() -> str:
    env = C.REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("LLAMA_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


_KEY = _api_key()
_HDR = {"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"}


# ── section-aware ini editing ─────────────────────────────────────────────────
def _section_bounds(text: str, section: str) -> tuple[int, int]:
    """Return [start, end) char offsets of a section's body in the ini text.

    section "*" matches the ``[*]`` header; any other name matches ``[name]``.
    The body runs from just after the header line to the next ``[`` header
    (or EOF).
    """
    header = re.escape(f"[{section}]")
    m = re.search(rf"^\s*{header}\s*$", text, re.M)
    if not m:
        raise KeyError(f"section [{section}] not found in models.ini")
    start = m.end()
    nxt = re.search(r"^\s*\[[^\]]+\]\s*$", text[start:], re.M)
    end = start + nxt.start() if nxt else len(text)
    return start, end


def set_params(params: dict) -> None:
    """Set ``key = value`` for each param inside its declared section.

    If the key already exists in the section it is replaced in place; otherwise
    it is appended at the end of the section body (so newly-introduced flags such
    as ``spec-draft-backend-sampling`` can be swept without a manual ini edit).
    """
    text = C.INI_PATH.read_text()
    for key, value in params.items():
        section = C.KEY_SECTION.get(key, "*")
        start, end = _section_bounds(text, section)
        body = text[start:end]
        pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.M)
        if pat.search(body):
            body = pat.sub(lambda m: m.group(1) + str(value), body, count=1)
        else:
            body = body.rstrip("\n") + f"\n{key:25s} = {value}\n"
        text = text[:start] + body + text[end:]
    C.INI_PATH.write_text(text)


def read_params(keys) -> dict:
    """Read current values of keys from their declared sections."""
    text = C.INI_PATH.read_text()
    out: dict[str, str | None] = {}
    for key in keys:
        section = C.KEY_SECTION.get(key, "*")
        try:
            start, end = _section_bounds(text, section)
        except KeyError:
            out[key] = None
            continue
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.*)$", text[start:end], re.M)
        out[key] = m.group(1).strip() if m else None
    return out


# ── container lifecycle ───────────────────────────────────────────────────────
def restart() -> None:
    subprocess.run(
        C.COMPOSE + ["up", "-d", "--force-recreate", "llama-server"],
        cwd=C.REPO_ROOT, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

def _server_crashed() -> bool:
    """True if the freshly-recreated container logged a fatal model-load error.

    force-recreate replaces the container, so its logs only cover the current
    config. An unsupported spec-type (e.g. ``ngram-mod`` with no draft context)
    aborts with ``GGML_ASSERT(ctx_dft)`` / ``loading error`` — detecting that lets
    us fail the config fast instead of waiting out the full health timeout.
    """
    try:
        out = subprocess.run(
            C.COMPOSE + ["logs", "--tail", "40", "llama-server"],
            cwd=C.REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return False
    blob = (out.stdout or "") + (out.stderr or "")
    return ("GGML_ASSERT" in blob) or ("loading error" in blob)

def wait_ready(timeout: int = C.HEALTH_TIMEOUT) -> float:
    """Block until the model actually serves a token. Returns seconds waited.

    Raises ``RuntimeError`` fast if the container logs a fatal model-load error
    (an unsupported config that can never serve), so the sweep can record it as
    non-viable instead of stalling for the full timeout.
    """
    t0 = time.time()
    # 1) http endpoint up
    while time.time() - t0 < timeout:
        if _server_crashed():
            raise RuntimeError("llama-server crashed during model load")
        try:
            urllib.request.urlopen(C.API_BASE + "/v1/models", timeout=5)
            break
        except Exception:
            time.sleep(3)
    else:
        raise TimeoutError("server http never came up")
    # 2) model loaded and able to decode
    probe = json.dumps({
        "model": C.MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }).encode()
    while time.time() - t0 < timeout:
        if _server_crashed():
            raise RuntimeError("llama-server crashed during model load")
        try:
            req = urllib.request.Request(
                C.API_BASE + "/v1/chat/completions", method="POST",
                headers=_HDR, data=probe)
            urllib.request.urlopen(req, timeout=30)
            return time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code in (500, 503):
                time.sleep(4)
                continue
            time.sleep(4)
        except Exception:
            time.sleep(4)
    raise TimeoutError("model never became ready")


# ── measurement ───────────────────────────────────────────────────────────────
def _stream_one(payload_path: Path) -> dict:
    """Stream a single completion; return timing + usage for that stream."""
    body = json.loads(payload_path.read_text())
    req = urllib.request.Request(
        C.API_BASE + "/v1/chat/completions", method="POST",
        headers=_HDR, data=json.dumps(body).encode())
    t_start = time.time()
    arrivals: list[float] = []
    ttft = None
    prompt_tokens = completion_tokens = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t_start
                    arrivals.append(now)
            if obj.get("usage"):
                prompt_tokens = obj["usage"].get("prompt_tokens")
                completion_tokens = obj["usage"].get("completion_tokens")

    n = len(arrivals)
    skip = C.WARMUP_SKIP
    if n > skip + 5:
        tg = (n - 1 - skip) / (arrivals[-1] - arrivals[skip])
    elif n >= 2:
        tg = (n - 1) / (arrivals[-1] - arrivals[0])
    else:
        tg = 0.0
    return {
        "tg": round(tg, 2),
        "ctok": completion_tokens or n,
        "ptok": prompt_tokens,
        "ttft": round(ttft or 0.0, 2),
    }


def _stat(xs: list[float]) -> tuple[float, float]:
    """Return (median, coefficient-of-variation %) for a list of samples."""
    s = sorted(xs)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    mean = sum(s) / n
    if mean == 0:
        return median, 0.0
    std = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    return median, std / mean * 100


def measure(payload_path: Path, concurrency: int = 1, repeats: int = 1) -> dict:
    """Measure decode throughput, repeating ``repeats`` times for reproducibility.

    Returns the *median* tg across repeats plus the spread (``tg_runs`` and the
    coefficient of variation ``tg_cv`` in %), so run-to-run variance is visible.
    For concurrency==1 each repeat is one stream; for >1 each repeat is a full
    parallel batch and ``tg`` is the aggregate (sum of per-stream) throughput.

    TTFT is taken from the *first* repeat only — that's the single cold-cache
    request; later repeats hit the server's prompt cache and would report a
    misleadingly tiny first-token latency.
    """
    repeats = max(1, repeats)
    runs = [
        _stream_one(payload_path) if concurrency == 1
        else _measure_parallel(payload_path, concurrency)
        for _ in range(repeats)
    ]

    tgs = [r["tg"] for r in runs]
    tg_med, tg_cv = _stat(tgs)
    cold = runs[0]
    out = {
        "tg": round(tg_med, 2),
        "tg_runs": [round(x, 2) for x in tgs],
        "tg_cv": round(tg_cv, 1),
        "ctok": cold["ctok"],
        "ptok": cold["ptok"],
        "ttft": cold["ttft"],              # cold-cache first-token latency
    }
    if concurrency > 1:
        out["tg_per_stream"] = cold.get("tg_per_stream", [])
    return out


def _measure_parallel(payload_path: Path, concurrency: int) -> dict:
    """Run ``concurrency`` identical streams in parallel (one batch).

    Returns the aggregate decode throughput (sum of per-stream tg) plus the
    per-stream tg list — the metric that matters for multi-slot serving.
    """
    results: list[dict] = [None] * concurrency  # type: ignore
    errors: list[BaseException | None] = [None] * concurrency
    threads = []

    def worker(i: int):
        try:
            results[i] = _stream_one(payload_path)
        except BaseException as e:  # noqa: BLE001 -- surfaced from main thread below
            errors[i] = e

    for i in range(concurrency):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    failed = [e for e in errors if e is not None]
    if failed:
        # A worker thread's exception is otherwise swallowed (it would leave a
        # None in results and crash later with a cryptic TypeError, losing all
        # sweep progress). Re-raise as RuntimeError so run_config's handler
        # records this config as non-viable (tg=0) and the sweep continues.
        raise RuntimeError(
            f"{len(failed)}/{concurrency} parallel streams failed: {failed[0]!r}"
        ) from failed[0]

    tgs = [r["tg"] for r in results]
    return {
        "tg": round(sum(tgs), 2),               # aggregate throughput
        "tg_per_stream": [round(x, 2) for x in tgs],
        "ctok": sum(r["ctok"] for r in results),
        "ptok": results[0]["ptok"],
        "ttft": round(max(r["ttft"] for r in results), 2),
    }
