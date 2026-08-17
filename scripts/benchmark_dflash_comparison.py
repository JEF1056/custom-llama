#!/usr/bin/env python3
"""
Benchmark comparison script for Qwen3.8-27B-heretic-ara:
Compares Native Autoregressive Decoding vs DFlash Speculative Decoding (draft_num_tokens=3).
Tests across multi-language coding, mathematics, and technical reasoning prompts.
"""

import json
import time
import urllib.request
import subprocess

PROMPTS = [
    ("Python Fibonacci", "Write a Python function to compute the Fibonacci sequence up to n elements with type annotations and docstring."),
    ("Rust Concurrency", "Implement a thread-safe in-memory cache in Rust using Arc and Mutex with get and insert methods."),
    ("C++ QuickSort", "Write an efficient in-place generic template QuickSort function in C++."),
    ("Math Reasoning", "A train travels from City A to City B at 60 mph and returns at 40 mph. What is the average speed for the entire round trip? Explain step-by-step."),
    ("Technical Explanation", "Explain the architectural difference between speculative decoding and standard autoregressive decoding in 2 concise paragraphs.")
]

NODES = ["ml-2", "ml-3"]


def start_server(node: str, enable_dflash: bool):
    print(f"\n>>> Starting server on {node} (DFlash Enabled: {enable_dflash})...", flush=True)
    dflash_args = "--draft-model jfan/Qwen3.8-27B-heretic-dflash --draft-kind dflash --draft-num-tokens 3" if enable_dflash else ""
    
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
    results = {node: {} for node in NODES}

    for node in NODES:
        print(f"\n--- Running Benchmark Prompts on {node} ({mode_name}) ---", flush=True)
        for task_name, prompt in PROMPTS:
            print(f"  > Testing [{task_name}]...", flush=True)
            payload = {
                "model": "/Users/jfan/.qwen/models/Qwen3.8-27B-heretic-ara-mxfp4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0,
                "stream": True,
            }
            try:
                t_start = time.time()
                req = urllib.request.Request(
                    f"http://{node}:8080/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                timings = {}
                tokens_count = 0
                with urllib.request.urlopen(req, timeout=180) as resp:
                    for line in resp:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: ") and not line.endswith("[DONE]"):
                            try:
                                chunk = json.loads(line[6:])
                                tokens_count += 1
                                if "timings" in chunk and chunk["timings"]:
                                    timings = chunk["timings"]
                            except Exception:
                                pass
                latency = time.time() - t_start
                dec_tps = timings.get("predicted_per_second", tokens_count / latency if latency > 0 else 0)
                pref_tps = timings.get("prompt_per_second", 0)
                draft_acc = (timings.get("draft_n_accepted", 0) / timings.get("draft_n", 1)) * 100 if timings.get("draft_n") else 0.0
                
                results[node][task_name] = {
                    "decode_tps": dec_tps,
                    "prefill_tps": pref_tps,
                    "latency": latency,
                    "draft_acc": draft_acc,
                    "accepted": timings.get("draft_n_accepted"),
                    "drafted": timings.get("draft_n")
                }
                print(f"    Decode: {dec_tps:.2f} tok/s | Prefill: {pref_tps:.1f} tok/s | Latency: {latency:.2f}s | Acc: {draft_acc:.1f}% ({timings.get('draft_n_accepted')}/{timings.get('draft_n')})", flush=True)
            except Exception as e:
                print(f"    Error on [{task_name}]: {e}", flush=True)
                results[node][task_name] = {"decode_tps": 0.0, "latency": 0.0, "draft_acc": 0.0}

    return results


def main():
    with_dflash = run_benchmark_for_mode(enable_dflash=True)
    without_dflash = run_benchmark_for_mode(enable_dflash=False)

    print("\n=======================================================", flush=True)
    print(" FINAL COMPARATIVE BENCHMARK SUMMARY (HIGH CONVERGENCE DFLASH)", flush=True)
    print("=======================================================\n", flush=True)

    for node in NODES:
        print(f"=================== NODE: {node.upper()} ===================", flush=True)
        print(f"{'Prompt Task':<24} | {'Native Decode':<14} | {'DFlash Decode':<14} | {'DFlash Acc %':<12} | {'Speedup':<8}")
        print("-" * 80)
        for task_name, _ in PROMPTS:
            nat = without_dflash.get(node, {}).get(task_name, {}).get("decode_tps", 0.0)
            dfl = with_dflash.get(node, {}).get(task_name, {}).get("decode_tps", 0.0)
            acc = with_dflash.get(node, {}).get(task_name, {}).get("draft_acc", 0.0)
            speedup = (dfl / nat) if nat > 0 else 0.0
            print(f"{task_name:<24} | {nat:>8.2f} tok/s   | {dfl:>8.2f} tok/s   | {acc:>7.1f}%      | {speedup:>5.2f}x")
        print()


if __name__ == "__main__":
    main()
