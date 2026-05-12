"""
title: Llama KV Cache Manager
description: >
  Pipe pipeline for Open WebUI. Routes completions to llama.cpp and manages
  per-conversation KV cache slot lifecycle:
    - Maps each chat_id to a fixed slot (round-robin over LLAMA_PARALLEL slots)
    - Restores the slot's KV cache before each request (via /slots/{id}?action=restore)
    - Injects id_slot + cache_prompt=true into the upstream body
    - Saves the slot after each response (non-blocking background thread)
  Slot-to-chat mappings survive pipeline reloads via a JSON state file.
author: custom
version: 0.2.0
"""

import json
import os
import threading
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Union

import requests
from pydantic import BaseModel

_STATE_FILE = Path("/app/pipelines/slot_state.json")


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    try:
        _STATE_FILE.write_text(json.dumps(data))
    except Exception:
        pass


class Pipeline:
    class Valves(BaseModel):
        LLAMA_BASE_URL: str = os.getenv(
            "LLAMA_BASE_URL", "http://llama-server:8080"
        )
        NUM_SLOTS: int = int(os.getenv("LLAMA_PARALLEL", "2"))
        SLOT_SAVE_ENABLED: bool = True

    def __init__(self):
        self.name = "Llama KV Cache"
        self.valves = self.Valves()
        self._lock = threading.Lock()
        # chat_id → slot index; persisted across reloads
        self._slot_map: dict[str, int] = _load_state()

    def get_models(self) -> list[dict]:
        """Proxy the model list from llama.cpp so Open WebUI shows real names."""
        try:
            r = requests.get(
                f"{self.valves.LLAMA_BASE_URL}/v1/models", timeout=5
            )
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception:
            return [{"id": "llama", "name": "Llama (loading…)"}]

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        chat_id: str = body.get("metadata", {}).get("chat_id", "")
        slot: Optional[int] = None

        if chat_id:
            slot = self._get_or_assign_slot(chat_id)
            self._restore_slot(slot)
            body["id_slot"] = slot
            body["cache_prompt"] = True

        # Strip Open WebUI internal keys that llama.cpp doesn't understand.
        body.pop("metadata", None)
        body.pop("user", None)

        stream: bool = body.get("stream", False)
        url = f"{self.valves.LLAMA_BASE_URL}/v1/chat/completions"
        r = requests.post(url, json=body, stream=stream, timeout=None)
        r.raise_for_status()

        if stream:
            return self._stream_and_save(r, slot)

        if slot is not None:
            self._save_slot_bg(slot)
        return r.json()

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def _get_or_assign_slot(self, chat_id: str) -> int:
        with self._lock:
            if chat_id not in self._slot_map:
                self._slot_map[chat_id] = (
                    len(self._slot_map) % self.valves.NUM_SLOTS
                )
                _save_state(self._slot_map)
            return self._slot_map[chat_id]

    def _restore_slot(self, slot: int) -> None:
        if not self.valves.SLOT_SAVE_ENABLED:
            return
        try:
            requests.post(
                f"{self.valves.LLAMA_BASE_URL}/slots/{slot}?action=restore",
                timeout=5,
            )
        except Exception:
            pass

    def _save_slot_bg(self, slot: int) -> None:
        """Fire-and-forget slot save so it never blocks the response."""
        if not self.valves.SLOT_SAVE_ENABLED:
            return

        def _do_save():
            try:
                requests.post(
                    f"{self.valves.LLAMA_BASE_URL}/slots/{slot}?action=save",
                    timeout=10,
                )
            except Exception:
                pass

        threading.Thread(target=_do_save, daemon=True).start()

    def _stream_and_save(
        self, response: requests.Response, slot: Optional[int]
    ) -> Iterator[bytes]:
        """Yield SSE bytes from llama.cpp, then save the slot."""
        try:
            yield from response.iter_content(chunk_size=None)
        finally:
            if slot is not None:
                self._save_slot_bg(slot)
