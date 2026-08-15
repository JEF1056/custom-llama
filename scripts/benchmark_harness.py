#!/bin/env python3
"""
=============================================================================
Centralized Multi-Node LLM Benchmark Harness
=============================================================================
Tests remote and local LLM server instances (Docker llama-server, MLX server)
across 3 prompt scale tiers:
  - Short  (2k tokens)
  - Medium (50k tokens)
  - Long   (125k tokens)

Evaluates:
  1. Time to First Token (TTFT)
  2. Prompt Processing Speed (Prefill Tokens/Sec)
  3. Decode Speed (Generation Tokens/Sec)
  4. Prefix Cache Performance (Pass 1 Cold vs Pass 2 Warm Cache hit delta)
=============================================================================
"""

import sys
import os
import time
import json
import argparse
import urllib.request
import urllib.error

# Approximate conversion: 1 token ~ 4 characters
CHAR_PER_TOKEN = 4

PROMPT_TIERS = {
    "2k": 2000,
    "50k": 50000,
    "125k": 125000,
}

BASE_PREFIX = (
    "You are a helpful AI assistant. Below is an extensive technical context document. "
    "Please read the context carefully and prepare to analyze it.\n\n--- BEGIN CONTEXT ---\n"
)

BASE_SUFFIX = "\n--- END CONTEXT ---\n\nQuestion: Summarize the key architectural principles presented in the text."


import random
import uuid

def generate_prompt(token_count: int, nonce: str = None) -> str:
    """Generate a prompt of roughly target token count with repeated technical text and unique nonce."""
    session_nonce = nonce or f"Session-{uuid.uuid4().hex[:8]}"
    prefix = (
        f"You are a helpful AI assistant. [Document Identifier: {session_nonce}]\n"
        "Below is an extensive technical context document. "
        "Please read the context carefully and prepare to analyze it.\n\n--- BEGIN CONTEXT ---\n"
    )
    sample_text = (
        "Distributed systems require robust state machine replication, consensus algorithms, "
        "and efficient key-value storage layouts. When designing high-throughput inference "
        "servers for large language models, memory bandwidth and KV-cache footprint dominate "
        "the execution bottleneck. Quantization techniques such as MXFP4, IQ4_KSS, and 4-bit "
        "block formats significantly reduce system VRAM usage while preserving model quality.\n"
    )
    target_chars = token_count * CHAR_PER_TOKEN
    prefix_len = len(prefix) + len(BASE_SUFFIX)
    needed_chars = max(100, target_chars - prefix_len)

    repeats = (needed_chars // len(sample_text)) + 1
    context = (sample_text * repeats)[:needed_chars]

    return prefix + context + BASE_SUFFIX


def reset_server_cache(server_url: str, api_key: str = None) -> bool:
    """Reset server APC / KV cache via /cache/reset (MLX) or /slots/{id}?action=erase (llama-server/Docker)."""
    base_url = server_url.rstrip("/")
    if base_url.endswith("/v1"):
        clean_base = base_url[:-3]
    else:
        clean_base = base_url

    reset_success = False

    # 1. Try MLX APC cache reset
    for path in ["/cache/reset", "/v1/cache/reset"]:
        target_url = clean_base + path
        req = urllib.request.Request(target_url, data=b"{}", headers={"Content-Type": "application/json"})
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    reset_success = True
        except Exception:
            pass

    # 2. Try llama-server (llama.cpp) slot cache erase
    for slot_id in range(4):
        for path in [f"/slots/{slot_id}?action=erase", f"/v1/slots/{slot_id}?action=erase"]:
            target_url = clean_base + path
            req = urllib.request.Request(target_url, data=b"{}", headers={"Content-Type": "application/json"})
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        reset_success = True
            except Exception:
                pass

    return reset_success


def get_default_model(server_url: str, api_key: str = None) -> str:
    """Fetch first model name from /v1/models endpoint."""
    url = f"{server_url.rstrip('/')}/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
    except Exception as e:
        print(f"[WARN] Failed to auto-detect model from {url}: {e}")

    return "default-model"


def run_single_benchmark(
    server_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    api_key: str = None,
    stream: bool = True
):
    """Executes a single completions request and measures prefill & decode timing."""
    url = f"{server_url.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)

    start_time = time.time()
    first_token_time = None
    last_token_time = None
    gen_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(content)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content") or delta.get("reasoning_content")
                        if text:
                            now = time.time()
                            if first_token_time is None:
                                first_token_time = now
                            last_token_time = now
                            gen_tokens += 1
                    except Exception:
                        pass
    except Exception as e:
        return {"error": str(e)}

    end_time = time.time()

    if first_token_time is None or gen_tokens == 0:
        return {"error": "No tokens received from server"}

    ttft = first_token_time - start_time  # Time to First Token (includes prefill)
    gen_duration = (last_token_time - first_token_time) if gen_tokens > 1 else (end_time - first_token_time)
    decode_tps = (gen_tokens - 1) / gen_duration if gen_duration > 0 and gen_tokens > 1 else gen_tokens / max(0.001, end_time - first_token_time)

    approx_prompt_tokens = len(prompt) // CHAR_PER_TOKEN
    prefill_tps = approx_prompt_tokens / max(0.001, ttft)

    return {
        "ttft_sec": round(ttft, 3),
        "prefill_tok_per_sec": round(prefill_tps, 1),
        "decode_tok_per_sec": round(decode_tps, 2),
        "gen_tokens": gen_tokens,
        "total_time_sec": round(end_time - start_time, 3),
        "prompt_tokens_est": approx_prompt_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Centralized LLM Benchmark Harness")
    parser.add_argument("--server", required=True, help="Base server URL (e.g. http://ml-2:8080/v1)")
    parser.add_argument("--model", help="Model name (auto-detected if omitted)")
    parser.add_argument("--api-key", help="Optional API key for authorization")
    parser.add_argument("--tiers", default="2k,50k,125k", help="Comma-separated tiers to run (2k,50k,125k)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Number of tokens to generate per test")
    parser.add_argument("--prefix-cache", action="store_true", help="Enable 2-pass prefix cache hit evaluation")

    args = parser.parse_args()

    model_name = args.model or get_default_model(args.server, args.api_key)
    tiers = [t.strip().lower() for t in args.tiers.split(",")]

    print("=============================================================================")
    print(f" LLM CENTRALIZED BENCHMARK HARNESS")
    print(f" Target Server : {args.server}")
    print(f" Target Model  : {model_name}")
    print(f" Tiers to Test : {', '.join(tiers)}")
    print(f" Prefix Cache  : {'Enabled (2-pass)' if args.prefix_cache else 'Disabled (1-pass)'}")
    print("=============================================================================\n")

    summary_results = {}

    for tier in tiers:
        if tier not in PROMPT_TIERS:
            print(f"[SKIP] Unknown tier: {tier}")
            continue

        target_tokens = PROMPT_TIERS[tier]
        print(f"--- Running Tier: {tier.upper()} (~{target_tokens} tokens) ---")
        prompt = generate_prompt(target_tokens)

        # Clear server cache prior to Pass 1 to ensure a true cold start
        if reset_server_cache(args.server, args.api_key):
            print("  [Cache Reset] Server cache cleared for cold pass.")

        # Pass 1: Cold Cache
        print(f"  [Pass 1 - Cold Cache] Submitting request...")
        res1 = run_single_benchmark(args.server, model_name, prompt, args.max_tokens, args.api_key)

        if "error" in res1:
            print(f"  [ERROR] Pass 1 Failed: {res1['error']}")
            summary_results[tier] = {"error": res1["error"]}
            continue

        print(f"    * TTFT (Prefill): {res1['ttft_sec']}s ({res1['prefill_tok_per_sec']} tok/s)")
        print(f"    * Decode Speed  : {res1['decode_tok_per_sec']} tok/s ({res1['gen_tokens']} tokens generated)")

        res2 = None
        if args.prefix_cache:
            time.sleep(1)
            print(f"  [Pass 2 - Warm Cache] Resubmitting identical prefix...")
            res2 = run_single_benchmark(args.server, model_name, prompt, args.max_tokens, args.api_key)

            if "error" not in res2:
                speedup = round(res1["ttft_sec"] / max(0.001, res2["ttft_sec"]), 2)
                print(f"    * TTFT (Warm)   : {res2['ttft_sec']}s ({res2['prefill_tok_per_sec']} tok/s)")
                print(f"    * Cache Speedup : {speedup}x TTFT reduction")

        summary_results[tier] = {
            "cold": res1,
            "warm": res2
        }
        print()

    # Print Markdown Table Report
    print("\n" + "=" * 78)
    print(" BENCHMARK SUMMARY REPORT")
    print("=" * 78)
    print(f" Server: {args.server} | Model: {model_name}\n")
    print(f"| Tier | Prompt Tokens | Cold TTFT | Prefill Speed | Decode Speed | Warm TTFT | Cache Speedup |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for tier, data in summary_results.items():
        if "error" in data:
            print(f"| {tier.upper()} | Error | - | - | - | - | - |")
            continue
        c = data["cold"]
        w = data.get("warm")
        warm_ttft_str = f"{w['ttft_sec']}s" if w else "N/A"
        speedup_str = f"{round(c['ttft_sec'] / max(0.001, w['ttft_sec']), 2)}x" if w else "N/A"

        print(
            f"| {tier.upper()} | ~{c['prompt_tokens_est']} | {c['ttft_sec']}s | "
            f"{c['prefill_tok_per_sec']} tok/s | {c['decode_tok_per_sec']} tok/s | "
            f"{warm_ttft_str} | {speedup_str} |"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
