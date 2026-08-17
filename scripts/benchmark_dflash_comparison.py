#!/usr/bin/env python3
"""
Benchmark comparison script for ml-2 and ml-3:
Runs a benchmark suite with and without DFlash speculative decoding.
Collects decode tok/s, prefill tok/s, latency, peak memory, and acceptance rate.
"""

import time
import json
import urllib.request
import subprocess

BENCH_PROMPTS = [
    ("Python Fibonacci", "Write a Python function to compute the Fibonacci sequence up to n elements with type annotations and docstring."),
    ("Rust Concurrency", "Implement a thread-safe in-memory cache in Rust using Arc and Mutex with get and insert methods."),
    ("C++ QuickSort", "Write an efficient in-place generic template QuickSort function in C++."),
    ("Math Reasoning", "A train travels from City A to City B at 60 mph and returns at 40 mph. What is the average speed for the entire round trip? Explain step-by-step."),
    ("Technical Explanation", "Explain the architectural difference between speculative decoding and standard autoregressive decoding in 2 concise paragraphs.")
]

NODES = ["ml-2", "ml-3"]


def start_server(node: str, enable_dflash: bool):
    print(f"\n>>> Starting server on {node} (DFlash Enabled: {enable_dflash})...", flush=True)
    dflash_args = "--draft-model jfan/Qwen3.8-27B-heretic-dflash --draft-kind dflash" if enable_dflash else ""
    
    script = f"""#!/usr/bin/env bash
MODEL_PATH="/Users/jfan/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4"
exec /opt/homebrew/bin/python3 -m mlx_vlm.server \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --model "/Users/jfan/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4" \\
    --trust-remote-code \\
    --kv-bits 4 \\
    --kv-quant-scheme uniform \\
    --kv-group-size 64 \\
    --prefill-step-size 2048 \\
    --max-kv-size 131072 \\
    {dflash_args}
"""
    subprocess.run(f"ssh {node} \"pkill -9 -f 'mlx_vlm.server' || true ; cat << 'EOF' > /tmp/run_bench_server.sh\n{script}\nEOF\nchmod +x /tmp/run_bench_server.sh ; /tmp/run_bench_server.sh > /tmp/bench_server.log 2>&1 &\"", shell=True)


def wait_for_server(node: str, timeout: int = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"http://{node}:8080/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"Server on {node} is READY ({int(time.time()-t0)}s).", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(3)
    print(f"Timeout waiting for server on {node}!", flush=True)
    return False


def run_benchmark_for_mode(enable_dflash: bool):
    mode_name = "WITH DFlash" if enable_dflash else "WITHOUT DFlash (Native AR)"
    print(f"\n=======================================================", flush=True)
    print(f" BENCHMARK RUN: {mode_name}", flush=True)
    print(f"=======================================================", flush=True)

    for node in NODES:
        start_server(node, enable_dflash)

    for node in NODES:
        if not wait_for_server(node):
            print(f"Skipping {node} due to startup failure.")
            continue

    time.sleep(5)
    results = {}

    for node in NODES:
        results[node] = []
        print(f"\n--- Running Benchmark Prompts on {node} ({mode_name}) ---", flush=True)
        for label, prompt in BENCH_PROMPTS:
            print(f"  > Testing [{label}]...", flush=True)
            payload = {
                "model": "/Users/jfan/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 128,
                "temperature": 0.0,
                "stream": False
            }
            req = urllib.request.Request(
                f"http://{node}:8080/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode())
                    elapsed = time.time() - t0
                    timings = data.get("timings", {})
                    pred_s = timings.get("predicted_per_second", 0)
                    prompt_s = timings.get("prompt_per_second", 0)
                    peak_mem = timings.get("peak_memory", 0)
                    draft_n = timings.get("draft_n", 0)
                    draft_accepted = timings.get("draft_n_accepted", 0)
                    accept_rate = (draft_accepted / draft_n * 100) if (draft_n and draft_n > 0) else 0.0

                    print(f"    Decode: {pred_s:.2f} tok/s | Prefill: {prompt_s:.1f} tok/s | Latency: {elapsed:.2f}s | Acc: {accept_rate:.1f}% ({draft_accepted}/{draft_n})", flush=True)
                    results[node].append({
                        "label": label,
                        "pred_tok_s": pred_s,
                        "prompt_tok_s": prompt_s,
                        "latency_s": elapsed,
                        "peak_mem_gb": peak_mem,
                        "draft_n": draft_n,
                        "draft_accepted": draft_accepted,
                        "accept_rate": accept_rate
                    })
            except Exception as e:
                print(f"    Error on [{label}]: {e}", flush=True)

    return results


def main():
    # 1. Run WITH DFlash
    dflash_results = run_benchmark_for_mode(enable_dflash=True)

    # 2. Run WITHOUT DFlash (Native)
    native_results = run_benchmark_for_mode(enable_dflash=False)

    print("\n=======================================================", flush=True)
    print(" FINAL COMPARATIVE BENCHMARK SUMMARY (CURRENT PRE-THINK DFLASH MODEL)", flush=True)
    print("=======================================================", flush=True)
    
    for node in NODES:
        print(f"\n=================== NODE: {node.upper()} ===================", flush=True)
        print(f"{'Prompt Task':<24} | {'Native Decode':<14} | {'DFlash Decode':<14} | {'DFlash Acc %':<12} | {'Speedup':<8}")
        print("-" * 80)
        df_list = dflash_results.get(node, [])
        nat_list = native_results.get(node, [])
        for i in range(len(df_list)):
            lbl = df_list[i]["label"]
            df_dec = df_list[i]["pred_tok_s"]
            nat_dec = nat_list[i]["pred_tok_s"] if i < len(nat_list) else 0
            acc = df_list[i]["accept_rate"]
            speedup = f"{df_dec / nat_dec:.2f}x" if nat_dec > 0 else "N/A"
            print(f"{lbl:<24} | {nat_dec:>7.2f} tok/s   | {df_dec:>7.2f} tok/s   | {acc:>6.1f}%      | {speedup:>6}")


if __name__ == "__main__":
    main()
