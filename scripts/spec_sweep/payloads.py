"""Reproducible prompt-payload generation.

Builds streaming chat-completion request bodies from in-repo corpora
(``calibration-data/wikitext-2-raw-test.txt`` for prose, the repo's own python
sources for code) so the sweep is fully reproducible from a clean checkout.

Each payload appends ``/no_think`` so the model emits real content/code rather
than spending the token budget on reasoning.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from . import config as C


def _load_prose() -> str:
    return C.PROSE_CORPUS.read_text(encoding="utf-8", errors="ignore")


def _load_code() -> str:
    parts: list[str] = []
    for pattern in C.CODE_GLOBS:
        for path in sorted(glob.glob(str(C.REPO_ROOT / pattern), recursive=True)):
            try:
                parts.append(Path(path).read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n\n".join(parts)


def _body(content: str) -> dict:
    return {
        "model": C.MODEL,
        "messages": [{"role": "user", "content": content + "\n/no_think"}],
        "max_tokens": C.MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.6,
        "top_p": 0.95,
    }


def _repeat_to(text: str, target_chars: int) -> str:
    if len(text) >= target_chars:
        return text[:target_chars]
    reps = (target_chars // len(text)) + 1
    return (text * reps)[:target_chars]


def build_all() -> dict[str, Path]:
    """(Re)generate every payload file. Returns {name: path}."""
    C.PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    prose = _load_prose()
    code = _load_code()

    specs = {
        "text_25k": ("prose", C.CTX_MID,
                     "The following is an encyclopedia excerpt. Read it, then continue "
                     "writing several new factual paragraphs in the same style.\n\n{corpus}"
                     "\n\nContinue with new related encyclopedic paragraphs:\n"),
        "text_160k": ("prose", C.CTX_LONG,
                      "The following is a long encyclopedia excerpt. Read it, then write "
                      "several new factual paragraphs continuing it.\n\n{corpus}"
                      "\n\nContinue with new related encyclopedic paragraphs:\n"),
        "text_90k": ("prose", C.CTX_SLOT,
                     "The following is an encyclopedia excerpt. Read it, then continue "
                     "writing several new factual paragraphs.\n\n{corpus}"
                     "\n\nContinue with new related encyclopedic paragraphs:\n"),
        "code_25k": ("code", C.CTX_MID,
                     "Continue implementing the following codebase. Output only code.\n\n"
                     "{corpus}\n\n# Continue the implementation below:\n"),
        "code_160k": ("code", C.CTX_LONG,
                      "Continue implementing the following large codebase. Output only code.\n\n"
                      "{corpus}\n\n# Continue the implementation below:\n"),
        "code_90k": ("code", C.CTX_SLOT,
                     "Continue implementing the following codebase. Output only code.\n\n"
                     "{corpus}\n\n# Continue the implementation below:\n"),
    }

    out: dict[str, Path] = {}
    for name, (kind, target_tok, template) in specs.items():
        if kind == "prose":
            corpus = _repeat_to(prose, int(target_tok * C.PROSE_CHARS_PER_TOK))
        else:
            corpus = _repeat_to(code, int(target_tok * C.CODE_CHARS_PER_TOK))
        body = _body(template.format(corpus=corpus))
        path = C.PAYLOAD_DIR / f"{name}.json"
        path.write_text(json.dumps(body))
        out[name] = path
    return out


def ensure() -> None:
    """Build payloads only if any are missing."""
    needed = ["text_25k", "text_160k", "text_90k", "code_25k", "code_160k", "code_90k"]
    if all((C.PAYLOAD_DIR / f"{n}.json").exists() for n in needed):
        return
    build_all()


if __name__ == "__main__":
    for name, path in build_all().items():
        body = json.loads(path.read_text())
        print(f"{name:10s} chars={len(body['messages'][0]['content']):>8d}  -> {path}")
