"""In-memory store for large tool outputs, enabling paginated reads.

MCP has no protocol-level pagination for tool-call results (only for *list*
operations). To let the model fetch outputs that would otherwise be truncated,
oversized results are stored here under an opaque handle and read back in
windows via the ``read_output`` tool.

Entries are evicted by TTL and by LRU once capacity is exceeded. State is
process-local, which is sufficient for this single-user, single-process server.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from src.config import settings


@dataclass
class _Entry:
    text: str
    source: str
    created: float
    last_access: float


class OutputStore:
    """Thread-safe TTL + LRU store for large textual tool outputs."""

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def _evict_locked(self) -> None:
        now = time.time()
        expired = [h for h, e in self._entries.items() if now - e.created > self.ttl]
        for h in expired:
            del self._entries[h]
        while len(self._entries) > self.max_entries:
            oldest = min(self._entries, key=lambda h: self._entries[h].last_access)
            del self._entries[oldest]

    def store(self, text: str, *, source: str = "output") -> str:
        """Store full text and return an opaque handle for later reads."""
        with self._lock:
            self._evict_locked()
            handle = "out_" + uuid.uuid4().hex[:12]
            now = time.time()
            self._entries[handle] = _Entry(text=text, source=source, created=now, last_access=now)
            return handle

    def read(self, handle: str, *, offset: int = 0, limit: int | None = None) -> dict:
        """Read a character window of a stored output.

        Returns a dict with status, the content slice, and pagination metadata.
        Unknown or expired handles return ``status="error"``.
        """
        limit = limit or settings.READ_OUTPUT_CHUNK_CHARS
        with self._lock:
            entry = self._entries.get(handle)
            if entry is not None and time.time() - entry.created > self.ttl:
                del self._entries[handle]
                entry = None
            if entry is None:
                return {
                    "status": "error",
                    "error": (
                        f"Unknown or expired handle '{handle}'. Handles live for "
                        f"{self.ttl}s; re-run the original tool to regenerate it."
                    ),
                }
            entry.last_access = time.time()
            text = entry.text
            source = entry.source

        total = len(text)
        offset = max(0, offset)
        if offset >= total:
            return {
                "status": "success",
                "handle": handle,
                "source": source,
                "offset": offset,
                "returned_chars": 0,
                "total_chars": total,
                "has_more": False,
                "next_offset": None,
                "content": "",
            }
        end = min(total, offset + limit)
        chunk = text[offset:end]
        has_more = end < total
        return {
            "status": "success",
            "handle": handle,
            "source": source,
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": total,
            "has_more": has_more,
            "next_offset": end if has_more else None,
            "content": chunk,
        }

    def attach(
        self,
        target: dict,
        key: str,
        full_text: str,
        *,
        source: str,
        inline_chars: int | None = None,
        preview: str | None = None,
    ) -> dict:
        """Place ``full_text`` into ``target[key]``, paginating when oversized.

        If ``full_text`` fits within ``inline_chars`` it is stored inline.
        Otherwise a preview (head slice, or the supplied ``preview``) is stored
        inline and a handle plus pagination hints are added under
        ``{key}_handle`` / ``{key}_total_chars`` / ``{key}_next_offset`` /
        ``{key}_hint`` so the model can call ``read_output`` for the remainder.
        """
        inline_chars = inline_chars or settings.OUTPUT_PREVIEW_CHARS
        total = len(full_text)
        if total <= inline_chars:
            target[key] = full_text
            return target

        if preview is not None:
            head = preview
            next_off = 0
        else:
            head_slice = full_text[:inline_chars]
            head = head_slice.rsplit(" ", 1)[0] if " " in head_slice else head_slice
            head = head + "…"
            next_off = len(head) - 1  # exclude the appended ellipsis

        handle = self.store(full_text, source=source)
        target[key] = head
        target[f"{key}_handle"] = handle
        target[f"{key}_total_chars"] = total
        target[f"{key}_next_offset"] = next_off
        target[f"{key}_hint"] = (
            f"Preview only — full {key} is {total} chars. "
            f'Call read_output(handle="{handle}", offset={next_off}) to continue.'
        )
        return target


output_store = OutputStore(
    ttl_seconds=settings.OUTPUT_STORE_TTL,
    max_entries=settings.OUTPUT_STORE_MAX_ENTRIES,
)
