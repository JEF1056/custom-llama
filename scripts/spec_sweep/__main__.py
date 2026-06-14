"""CLI entry point for the spec-decode / context-parallel sweep.

    python -m scripts.spec_sweep run          # full staged sweep (resumable)
    python -m scripts.spec_sweep payloads     # (re)build prompt payloads only
    python -m scripts.spec_sweep restore      # restore models.ini from backup
    python -m scripts.spec_sweep status       # show progress / decisions so far
    python -m scripts.spec_sweep reset        # clear state so the next run starts fresh

Run from the repo root: ``cd custom-llama && python -m scripts.spec_sweep run``.
"""
from __future__ import annotations

import json
import sys

from . import config as C
from . import harness, payloads
from .runner import Sweep


def _status() -> None:
    if not C.STATE_JSON.exists():
        print("no sweep state yet.")
        return
    state = json.loads(C.STATE_JSON.read_text())
    done = state.get("completed", {})
    print(f"completed configs: {len(done)}")
    for tag, info in done.items():
        print(f"  {tag:16s} {info.get('elapsed', '?')}s")
    print("decisions:", json.dumps(state.get("decisions", {}), indent=2))


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd == "payloads":
        for name, path in payloads.build_all().items():
            body = json.loads(path.read_text())
            print(f"{name:10s} chars={len(body['messages'][0]['content']):>8d} -> {path}")
    elif cmd == "restore":
        if C.INI_BACKUP.exists():
            C.INI_PATH.write_text(C.INI_BACKUP.read_text())
            print(f"restored models.ini from {C.INI_BACKUP}")
        else:
            print("no backup found.")
            return 1
    elif cmd == "status":
        _status()
    elif cmd == "reset":
        removed = []
        for p in (C.STATE_JSON, C.RESULTS_CSV):
            if p.exists():
                p.unlink()
                removed.append(p.name)
        print("reset:", ", ".join(removed) if removed else "nothing to remove")
        print("(models.ini left as-is; run `restore` to revert it to the pre-sweep backup)")
    elif cmd == "run":
        Sweep().run()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
