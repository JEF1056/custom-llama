#!/usr/bin/env python3
"""Benchmark long prompt (100k) across all 3 machines."""
import json, time, httpx, sys
sys.path.insert(0, '/home/jfan/custom-llama')
from benchmark_openai import generate_context, SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS

ENDPOINTS = [
    ('localhost (llamacpp)', 'http://localhost:8080/v1/chat/completions', '/models/qwen3.6-35b'),
    ('ml-2 (mlx)',         'http://ml-2:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
    ('ml-3 (mlx)',         'http://ml-3:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
]

TASK = {
    "desc": "Long (100k) — full codebase understanding",
    "prompt": "I will provide a very large Python codebase for a web application. After reading all of it, answer specific questions about the architecture and fix a reported bug. Specifically: (1) Identify the main entry point and explain the request flow. (2) Find any potential race conditions or concurrency issues. (3) Suggest improvements to the error handling strategy.",
    "target_tokens": 100000,
}

target_tokens = TASK["target_tokens"]
context = generate_context(target_tokens, seed="long")
actual_tokens = len(context) // 4
print(f"Context: ~{actual_tokens:,} tokens ({len(context):,} chars)")

msg = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': f"{TASK['prompt']}\n\n---\n\nHere is the codebase context:\n\n```\n{context}\n```"},
]

print()
print(f'{"Machine":<22} {"Time(s)":>8} {"P-tok/s":>10} {"D-tok/s":>10} {"Prompt":>10} {"Decode":>10}')
print(f'{"_"*22} {"_"*8} {"_"*10} {"_"*10} {"_"*10} {"_"*10}')

results = {}
for name, url, model in ENDPOINTS:
    print(f"\n-> {name}...", end=" ", flush=True)
    try:
        c = httpx.Client(timeout=600)
        t0 = time.time()
        r = c.post(url, json={
            'model': model,
            'messages': msg,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS,
            'stream': False,
        })
        dt = time.time() - t0
        r.raise_for_status()
        d = r.json()
        u = d.get('usage', {})
        tm = d.get('timings', {})
        pt, ct = u.get('prompt_tokens', 0), u.get('completion_tokens', 0)
        pps = tm.get('prompt_per_second', 0)
        dps = tm.get('predicted_per_second', 0) or (ct/dt if dt > 0 else 0)
        print(f"OK ({dt:.1f}s)")
        print(f'{name:<22} {dt:>8.1f} {pps:>10.1f} {dps:>10.1f} {pt:>10,} {ct:>10,}')
        results[name] = {
            'elapsed_s': round(dt, 1),
            'prompt_tokens': pt,
            'completion_tokens': ct,
            'prompt_tok_per_sec': round(pps, 1),
            'decode_tok_per_sec': round(dps, 1),
        }
        c.close()
    except Exception as e:
        print(f"FAILED ({e})")
        results[name] = {'error': str(e)}

# Save results
with open('/tmp/bench_long_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to /tmp/bench_long_results.json")
