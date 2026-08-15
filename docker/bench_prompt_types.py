#!/usr/bin/env python3
"""Benchmark short-context and long-context (130k+ token) generation across
prompt types (code, creative, QA, summarization) against a llama-server.

Usage: python3 bench_prompt_types.py [--base http://localhost:8081] [--long] [--short]
"""
import argparse
import json
import time
import urllib.request

DEFAULT_BASE = "http://localhost:8081"
N_PREDICT = 256

# --- Long-context filler (distinct per "type" so the model isn't just
# repeating an identical pattern across runs) -------------------------------

def _padded(paragraph_fmt, target_chars):
    chunks = []
    i = 0
    total = 0
    while total < target_chars:
        c = paragraph_fmt.format(n=i)
        chunks.append(c)
        total += len(c)
        i += 1
    return "".join(chunks)

CODE_PARA = (
    "def process_record_{n}(record):\n"
    "    # Validate and normalize incoming record {n} before persisting.\n"
    "    if not record.get('id'):\n"
    "        raise ValueError('missing id for record {n}')\n"
    "    record['checksum'] = compute_checksum(record, salt={n})\n"
    "    return record\n\n"
)

STORY_PARA = (
    "Chapter {n}: The travelers pressed on through the fog-laden valley, each "
    "footstep echoing against the stone walls that rose on either side. Elior "
    "recalled the old warning about the {n}th bridge, and wondered whether the "
    "map they carried still told the truth. "
)

FACTS_PARA = (
    "Entry {n}: The city council recorded {n} new permits this quarter, of "
    "which a fraction were commercial and the remainder residential. Historical "
    "records from year {n} indicate similar seasonal patterns in prior decades. "
)

DOC_PARA = (
    "Section {n}. In the study of complex systems, researchers observe that "
    "emergent behavior arises from the interaction of many simple components, "
    "each following local rules without central coordination. Consider a "
    "network of agents indexed by position {n}, exchanging signals across "
    "noisy channels, adapting internal state, and converging toward a shared "
    "equilibrium that no single agent could have predicted in advance. "
)

PROMPT_TYPES = {
    "code": {
        "short": (
            "Write a Python function `is_palindrome(s: str) -> bool` that "
            "ignores case and non-alphanumeric characters."
        ),
        "long_filler": CODE_PARA,
        "long_suffix": (
            "\n\nBriefly explain what `compute_checksum` most likely does, "
            "based on the usage pattern above, in one sentence."
        ),
    },
    "creative": {
        "short": (
            "Write a two-sentence opening line for a fantasy short story about "
            "a lighthouse keeper who discovers the sea is retreating forever."
        ),
        "long_filler": STORY_PARA,
        "long_suffix": (
            "\n\nBriefly describe, in one sentence, the mood established by "
            "the passage above."
        ),
    },
    "qa": {
        "short": (
            "What causes tides on Earth? Answer in two sentences."
        ),
        "long_filler": FACTS_PARA,
        "long_suffix": (
            "\n\nBased on the entries above, what general seasonal trend is "
            "described? Answer in one sentence."
        ),
    },
    "summarization": {
        "short": (
            "Summarize in one sentence: Complex systems exhibit emergent "
            "behavior from the interaction of simple, locally-governed "
            "components without central coordination."
        ),
        "long_filler": DOC_PARA,
        "long_suffix": (
            "\n\nBriefly summarize the passage above in one sentence."
        ),
    },
}

LONG_TARGET_CHARS = 750000  # ~130k+ tokens even for low tokens-per-char prose


def post(base, path, payload, timeout=3600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_one(base, label, prompt, n_predict=N_PREDICT):
    tok = post(base, "/tokenize", {"content": prompt})
    n_tok = len(tok["tokens"]) if isinstance(tok, dict) else len(tok)

    t0 = time.time()
    res = post(base, "/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": False,
    })
    wall = time.time() - t0

    tm = res.get("timings", {})
    result = {
        "label": label,
        "prompt_tokens": n_tok,
        "prompt_ms": round(tm.get("prompt_ms", 0), 1),
        "prompt_tps": round(tm.get("prompt_per_second", 0), 2),
        "predicted_n": tm.get("predicted_n"),
        "predicted_ms": round(tm.get("predicted_ms", 0), 1),
        "predicted_tps": round(tm.get("predicted_per_second", 0), 2),
        "draft_n": tm.get("draft_n"),
        "draft_n_accepted": tm.get("draft_n_accepted"),
        "wall_s": round(wall, 1),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--types", default="code,creative,qa,summarization")
    args = ap.parse_args()
    if not args.short and not args.long:
        args.short = args.long = True

    types = args.types.split(",")
    results = []

    if args.short:
        print("=== SHORT CONTEXT ===")
        for t in types:
            spec = PROMPT_TYPES[t]
            results.append(run_one(args.base, f"short/{t}", spec["short"]))

    if args.long:
        print("=== LONG CONTEXT (130k+ tokens) ===")
        for t in types:
            spec = PROMPT_TYPES[t]
            body = _padded(spec["long_filler"], LONG_TARGET_CHARS)
            prompt = body + spec["long_suffix"]
            results.append(run_one(args.base, f"long/{t}", prompt))

    print("=== SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
