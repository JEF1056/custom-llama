#!/usr/bin/env python3
"""
=============================================================================
Centralized Concurrent Multi-Node LLM Benchmark Harness
=============================================================================
Tests multiple remote and local LLM servers concurrently using a ThreadPoolExecutor
across 3 prompt scale tiers:
  - 2k   (~2,000 tokens)
  - 50k  (~50,000 tokens)
  - 125k (~125,000 tokens)

Evaluates:
  1. Time to First Token (TTFT / Prefill)
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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Approximate conversion: 1 token ~ 4 characters
CHAR_PER_TOKEN = 4

PROMPT_TIERS = {
    "2k": 2000,
    "50k": 50000,
    "125k": 125000,
}

BASE_SUFFIX = "\n--- END CONTEXT ---\n\nQuestion: Summarize the key architectural principles presented in the text."


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
        "quantized KV cache blocks minimize memory bus pressure and maximize token throughput.\n"
    )
    target_chars = token_count * CHAR_PER_TOKEN
    prefix_len = len(prefix) + len(BASE_SUFFIX)
    needed_chars = max(100, target_chars - prefix_len)

    repeats = (needed_chars // len(sample_text)) + 1
    context = (sample_text * repeats)[:needed_chars]

    return prefix + context + BASE_SUFFIX


def get_default_model(server_url: str, api_key: str = None) -> str:
    """Fetch the active model name from /v1/models endpoint."""
    url = f"{server_url.rstrip('/')}/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "data" in data and len(data["data"]) > 0:
                # Prefer explicitly loaded models
                for m in data["data"]:
                    mid = m["id"]
                    if "mxfp4" in mid or "qwen" in mid.lower():
                        return mid
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
        with urllib.request.urlopen(req, timeout=1200) as response:
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

    ttft = first_token_time - start_time
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


def benchmark_host(server_url: str, model: str, tiers: list, max_tokens: int, prefix_cache: bool, api_key: str):
    """Run all prompt tiers sequentially on a single host."""
    clean_url = server_url.rstrip("/")
    detected_model = model or get_default_model(clean_url, api_key)
    
    print(f"[{clean_url}] Starting benchmark (Model: {detected_model})...", flush=True)
    results = {}

    for tier in tiers:
        if tier not in PROMPT_TIERS:
            continue
        target_tokens = PROMPT_TIERS[tier]
        prompt = generate_prompt(target_tokens)

        # Pass 1: Cold Cache
        print(f"[{clean_url}] Running Tier {tier.upper()} Cold Pass...", flush=True)
        res1 = run_single_benchmark(clean_url, detected_model, prompt, max_tokens, api_key)
        if "error" in res1:
            print(f"[{clean_url}] Tier {tier.upper()} Cold Failed: {res1['error']}", flush=True)
            results[tier] = {"error": res1["error"]}
            continue

        print(f"[{clean_url}] Tier {tier.upper()} Cold: TTFT={res1['ttft_sec']}s ({res1['prefill_tok_per_sec']} tok/s), Decode={res1['decode_tok_per_sec']} tok/s", flush=True)

        res2 = None
        if prefix_cache:
            time.sleep(1)
            print(f"[{clean_url}] Running Tier {tier.upper()} Warm Pass...", flush=True)
            res2 = run_single_benchmark(clean_url, detected_model, prompt, max_tokens, api_key)
            if "error" not in res2:
                speedup = round(res1["ttft_sec"] / max(0.001, res2["ttft_sec"]), 2)
                print(f"[{clean_url}] Tier {tier.upper()} Warm: TTFT={res2['ttft_sec']}s ({speedup}x speedup)", flush=True)

        results[tier] = {
            "cold": res1,
            "warm": res2
        }

    return clean_url, detected_model, results


def main():
    parser = argparse.ArgumentParser(description="Centralized Parallel Multi-Node LLM Benchmark Harness")
    parser.add_argument("--servers", "-s", nargs="+", default=["http://ml-1-wsl:8080/v1", "http://ml-2:8080/v1", "http://ml-3:8080/v1"],
                        help="List of server URLs (e.g. http://ml-1-wsl:8080/v1 http://ml-2:8080/v1)")
    parser.add_argument("--models", "-m", nargs="*", default=[], help="Optional list of model names corresponding to servers")
    parser.add_argument("--api-key", help="Optional API key")
    parser.add_argument("--tiers", default="2k,50k,125k", help="Comma-separated tiers to run (2k,50k,125k)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Number of tokens to generate per test")
    parser.add_argument("--prefix-cache", action="store_true", default=True, help="Enable 2-pass prefix cache evaluation")
    parser.add_argument("--output-json", default="benchmark_results.json", help="Path to save raw results JSON")

    args = parser.parse_args()
    tiers = [t.strip().lower() for t in args.tiers.split(",")]

    print("=============================================================================")
    print(" CONCURRENT MULTI-NODE BENCHMARK HARNESS")
    print(f" Target Servers: {', '.join(args.servers)}")
    print(f" Tiers to Test : {', '.join(tiers)}")
    print(f" Prefix Cache  : {'Enabled (2-pass)' if args.prefix_cache else 'Disabled (1-pass)'}")
    print("=============================================================================\n")

    host_tasks = []
    for i, s_url in enumerate(args.servers):
        m_name = args.models[i] if i < len(args.models) else None
        host_tasks.append((s_url, m_name))

    all_results = {}
    with ThreadPoolExecutor(max_workers=len(host_tasks)) as executor:
        futures = {
            executor.submit(benchmark_host, s_url, m_name, tiers, args.max_tokens, args.prefix_cache, args.api_key): s_url
            for s_url, m_name in host_tasks
        }
        for future in as_completed(futures):
            s_url = futures[future]
            try:
                clean_url, detected_model, results = future.result()
                all_results[clean_url] = {
                    "model": detected_model,
                    "tiers": results
                }
            except Exception as e:
                print(f"[{s_url}] Execution error: {e}")
                all_results[s_url] = {"error": str(e)}

    # Save to JSON
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Benchmark results saved to {args.output_json}")

    # Print Formatted Report
    print("\n" + "=" * 90)
    print(" PARALLEL BENCHMARK COMPARISON REPORT")
    print("=" * 90)
    print(f"| Host / Server | Tier | Prompt Tokens | Cold TTFT | Prefill Speed | Decode Speed | Warm TTFT | Cache Speedup |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for s_url, h_data in all_results.items():
        if "error" in h_data:
            print(f"| {s_url} | Error | - | - | - | - | - | - |")
            continue
        for tier, t_data in h_data.get("tiers", {}).items():
            if "error" in t_data:
                print(f"| {s_url} | {tier.upper()} | Error | - | - | - | - | - |")
                continue
            c = t_data["cold"]
            w = t_data.get("warm")
            warm_ttft_str = f"{w['ttft_sec']}s" if w else "N/A"
            speedup_str = f"{round(c['ttft_sec'] / max(0.001, w['ttft_sec']), 2)}x" if w else "N/A"
            print(
                f"| {s_url} | {tier.upper()} | ~{c['prompt_tokens_est']} | {c['ttft_sec']}s | "
                f"{c['prefill_tok_per_sec']} tok/s | {c['decode_tok_per_sec']} tok/s | "
                f"{warm_ttft_str} | {speedup_str} |"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
