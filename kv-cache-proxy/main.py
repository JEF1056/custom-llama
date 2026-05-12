"""
KV Cache Proxy — OpenAI-compatible reverse proxy for Roo Code (tailnet access).

Adds per-session KV cache slot pinning to any llama.cpp server transparently:
  - Reads X-Session-ID request header to identify a session
  - Assigns the session a fixed llama.cpp slot (round-robin over NUM_SLOTS)
  - Restores the slot's KV cache before forwarding the request
  - Injects id_slot + cache_prompt=true into the request body
  - Saves the slot after the response completes (non-blocking)

Roo Code configuration:
  Base URL : http://<TS_HOSTNAME>.<tailnet>.ts.net:8181/v1
  API Key  : (any non-empty string, or leave blank)
  Custom headers:
    X-Session-ID: roo-<workspace-name>
"""

import asyncio
import json
import os
import threading
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI(title="KV Cache Proxy")

LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://llama-server:8080").rstrip("/")
NUM_SLOTS = int(os.getenv("NUM_SLOTS", "2"))
PORT = int(os.getenv("PORT", "8181"))
SLOT_SAVE_ENABLED = os.getenv("SLOT_SAVE_ENABLED", "true").lower() == "true"

_slot_map: dict[str, int] = {}
_lock = threading.Lock()


def get_or_assign_slot(session_id: str) -> int:
    with _lock:
        if session_id not in _slot_map:
            _slot_map[session_id] = len(_slot_map) % NUM_SLOTS
        return _slot_map[session_id]


async def restore_slot(slot: int) -> None:
    if not SLOT_SAVE_ENABLED:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(f"{LLAMA_BASE_URL}/slots/{slot}?action=restore")
    except Exception:
        pass


async def save_slot(slot: int) -> None:
    if not SLOT_SAVE_ENABLED:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"{LLAMA_BASE_URL}/slots/{slot}?action=save")
    except Exception:
        pass


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
async def proxy(request: Request, path: str):
    session_id = request.headers.get("x-session-id", "")
    raw_body = await request.body()
    body = raw_body
    stream = False
    slot: int | None = None

    # Only apply slot management on chat completions with a session header.
    if session_id and path.lstrip("/") == "v1/chat/completions" and raw_body:
        try:
            data = json.loads(raw_body)
            stream = data.get("stream", False)
            slot = get_or_assign_slot(session_id)
            await restore_slot(slot)
            data["id_slot"] = slot
            data["cache_prompt"] = True
            body = json.dumps(data).encode()
        except Exception:
            pass

    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "x-session-id"}
    }
    upstream_url = f"{LLAMA_BASE_URL}/{path.lstrip('/')}"
    query = dict(request.query_params)

    if stream:

        async def stream_and_save() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream(
                    request.method,
                    upstream_url,
                    headers=forward_headers,
                    content=body,
                    params=query,
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            if slot is not None:
                await save_slot(slot)

        return StreamingResponse(
            stream_and_save(), media_type="text/event-stream"
        )

    async with httpx.AsyncClient(timeout=None) as c:
        r = await c.request(
            request.method,
            upstream_url,
            headers=forward_headers,
            content=body,
            params=query,
        )

    if slot is not None:
        asyncio.create_task(save_slot(slot))

    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=dict(r.headers),
        media_type=r.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
