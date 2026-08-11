#!/usr/bin/env python3
"""Benchmark three OpenAI-compatible machines with 3 prompts per length."""
import json, time, httpx, sys
from pathlib import Path

sys.path.insert(0, '/home/jfan/custom-llama')
from benchmark_openai import generate_context, SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS, count_tokens_approx

ENDPOINTS = [
    ('localhost (llamacpp)', 'http://localhost:8080/v1/chat/completions', '/models/qwen3.6-35b'),
    ('ml-2 (mlx)',         'http://ml-2:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
    ('ml-3 (mlx)',         'http://ml-3:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
]

# 3 different prompts per length
PROMPTS = {
    "short": [
        "Refactor this CSV parser to a class-based approach with error handling and type hints. Write a CsvParser class with proper validation. Task A.",
        "Given a list of dictionaries representing user records, write a function to deduplicate by email while preserving the most recent entry. Task B.",
        "Convert this procedural data processing script to use a pipeline pattern with generator functions for memory efficiency. Task C.",
    ],
    "medium": [
        "Review this Python REST API codebase and identify bugs, then provide fixes. Focus on error handling and edge cases. Analysis 1.",
        "Analyze this codebase for potential security vulnerabilities. Identify SQL injection, XSS, and auth issues with specific line references. Audit 2.",
        "This codebase has a reported memory leak. Profile the code, identify the leak source, and provide a fix with explanation. Debug 3.",
    ],
    "long": [
        "After reading this full codebase: (1) Identify the main entry point and explain the request flow. (2) Find race conditions. (3) Suggest error handling improvements. Review A.",
        "Review this complete web app: (1) Explain the architecture and data flow. (2) Identify testing gaps. (3) Propose a migration strategy from sync to async. Review B.",
        "Analyze this codebase end-to-end: (1) Document the module dependencies. (2) Find performance bottlenecks. (3) Suggest refactoring priorities. Review C.",
    ],
}

TASKS = {
    "short": {"target_tokens": 4000, "max_tokens": 2048},
    "medium": {"target_tokens": 40000, "max_tokens": 2048},
    "long": {"target_tokens": 100000, "max_tokens": 2048},
}


def run_task(machine_name, url, model, task_size, prompt_idx, context, user_prompt, max_tokens):
    """Run a single benchmark task on one machine."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_prompt}\n\nContext:\n\n```\n{context}\n```"},
    ]
    try:
        c = httpx.Client(timeout=600)
        t0 = time.time()
        resp = c.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        timings = data.get("timings", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        predicted_per_sec = timings.get("predicted_per_second", 0)
        prompt_ms = timings.get("prompt_ms", 0)
        
        # Calculate actual prompt processing speed from raw timing
        # Server's prompt_per_second is misleading due to KV cache
        if prompt_ms > 0 and prompt_tokens > 0:
            actual_prompt_per_sec = prompt_tokens / (prompt_ms / 1000.0)
        else:
            actual_prompt_per_sec = predicted_per_sec if predicted_per_sec > 0 else (prompt_tokens / elapsed if elapsed > 0 else 0)
        
        actual_dps = predicted_per_sec if predicted_per_sec > 0 else (completion_tokens / elapsed if elapsed > 0 else 0)

        return {
            "elapsed_s": round(elapsed, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "decode_tok_per_sec": round(actual_dps, 1),
            "gpu_prompt_tok_per_sec": round(actual_prompt_per_sec, 1),
            "effective_prompt_tok_per_sec": round(prompt_tokens / elapsed, 1) if elapsed > 0 else 0,
            "prompt_ms": round(prompt_ms, 1),
            "success": True,
        }
    except Exception as e:
        return {
            "elapsed_s": round(time.time() - t0, 1) if 't0' in dir() else 0,
            "error": str(e),
            "success": False,
        }


def run_benchmark():
    all_results = {}

    for size_name, task_info in TASKS.items():
        target_tokens = task_info["target_tokens"]
        max_tokens = task_info["max_tokens"]
        prompt_list = PROMPTS[size_name]

        print(f"\n{'='*80}")
        print(f"  BENCHMARK: {size_name.upper()} ({target_tokens:,} token context)")
        print(f"{'='*80}")

        for prompt_idx, user_prompt in enumerate(prompt_list):
            print(f"\n  Prompt {prompt_idx + 1}/3: {user_prompt[:80]}...")

            for machine_name, url, model in ENDPOINTS:
                context = generate_context(target_tokens, seed=f"{size_name}_p{prompt_idx}_{machine_name.replace(' ', '_')}")
                actual_tokens = count_tokens_approx(context)
                print(f"  Context: ~{actual_tokens:,} tokens")
                print(f"    -> {machine_name}...", end=" ", flush=True)
                result = run_task(machine_name, url, model, size_name, prompt_idx, context, user_prompt, max_tokens)
                all_results.setdefault(size_name, {}).setdefault(prompt_idx, {})[machine_name] = result

                if result.get("success"):
                    print(f"OK ({result['elapsed_s']}s, {result['decode_tok_per_sec']} tok/s, GPU P:{result['gpu_prompt_tok_per_sec']} tok/s)")
                else:
                    print(f"FAIL ({result.get('error', 'unknown')[:40]})")

    # Print summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")

    for size_name in TASKS:
        print(f"\n  [{size_name.upper()} CONTEXT]")
        print(f'  {"Machine":<22} {"Avg Time":>10} {"GPU P-tok/s":>12} {"Eff P-tok/s":>12} {"Avg Prompt":>12} {"Avg Decode":>12}')
        print(f'  {"_"*22} {"_"*10} {"_"*12} {"_"*12} {"_"*12} {"_"*12}')

        for machine_name, _, _ in ENDPOINTS:
            values = []
            for prompt_idx in range(len(PROMPTS[size_name])):
                result = all_results.get(size_name, {}).get(prompt_idx, {}).get(machine_name, {})
                if result.get("success"):
                    values.append(result)

            if values:
                avg_time = sum(v["elapsed_s"] for v in values) / len(values)
                avg_dps = sum(v["decode_tok_per_sec"] for v in values) / len(values)
                avg_gpu_pts = sum(v.get("gpu_prompt_tok_per_sec", 0) for v in values) / len(values)
                avg_eff_pts = sum(v.get("effective_prompt_tok_per_sec", 0) for v in values) / len(values)
                avg_prompt_tok = sum(v["prompt_tokens"] for v in values) / len(values)
                avg_decode_tok = sum(v["completion_tokens"] for v in values) / len(values)
                print(f"  {machine_name:<22} {avg_time:>10.1f} {avg_gpu_pts:>12.1f} {avg_eff_pts:>12.1f} {avg_prompt_tok:>12.0f} {avg_decode_tok:>12.0f}")
            else:
                print(f"  {machine_name:<22} {'ERROR':>10}")

    # Save detailed results
    results_path = Path("/tmp/benchmark_results_v3.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Detailed results saved to {results_path}")

    # Save results
    results_path = Path("/tmp/benchmark_results_v2.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_benchmark()
