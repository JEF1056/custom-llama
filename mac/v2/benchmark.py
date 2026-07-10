import sys
import os
import time
import subprocess
import json
import urllib.request
import urllib.error

LOCAL_MODEL_PATH = "/Users/jfan/Documents/Qwen3.6-35B-A3B-MLX-4bit"
DRAFT_MODEL_PATH = "z-lab/Qwen3.6-35B-A3B-DFlash"

BENCHMARK_JSON = "/Users/jfan/Documents/vllm_benchmark.json"

def kill_port_8000():
    try:
        pid = subprocess.check_output(["lsof", "-ti", ":8000"]).decode("utf-8").strip()
        if pid:
            for p in pid.split():
                subprocess.call(["kill", "-9", p])
    except Exception:
        pass

def wait_for_server(timeout=120):
    start = time.time()
    payload = {
        "model": LOCAL_MODEL_PATH,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0
    }

    print("[BENCHMARK] Waiting for server model to load and respond...", end="", flush=True)
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8000/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    print(" Ready!", flush=True)
                    return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print(" Timeout!", flush=True)
    return False

def run_benchmark_for_config(draft_block_size):
    print(f"\n[BENCHMARK] Starting server with --draft-block-size {draft_block_size}...", flush=True)
    kill_port_8000()
    time.sleep(2)

    env = os.environ.copy()

    cmd = [
        "mlx_vlm.server",
        "--model", LOCAL_MODEL_PATH,
        "--draft-model", DRAFT_MODEL_PATH,
        "--kv-bits", "4",
        "--kv-quant-scheme", "turboquant",
        "--draft-block-size", str(draft_block_size),
        "--enable-thinking",
        "--thinking-budget", "2048",
        "--thinking-start-token", "<think>",
        "--thinking-end-token", "</think>",
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        if not wait_for_server():
            print(f"[BENCHMARK] Error: Server failed to start for draft-block-size={draft_block_size}", flush=True)
            return None

        print(f"[BENCHMARK] Server is ready. Sending coding task request...", flush=True)

        payload = {
            "model": LOCAL_MODEL_PATH,
            "messages": [
                {"role": "user", "content": "Write a highly optimized Python function to compute the longest common subsequence of two strings."}
            ],
            "max_tokens": 150,
            "temperature": 0.0,
            "stream": True
        }

        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        first_token_time = None
        tokens_count = 0

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        content = line[6:]
                        if content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(content)
                            delta = chunk["choices"][0]["delta"]
                            has_content = "content" in delta and delta["content"]
                            has_reasoning = "reasoning_content" in delta and delta["reasoning_content"]
                            if has_content or has_reasoning:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                tokens_count += 1
                        except Exception:
                            pass
            end_time = time.time()

            if tokens_count > 0 and first_token_time is not None:
                duration = end_time - first_token_time
                tokens_per_second = tokens_count / duration
                print(f"[BENCHMARK] Generated {tokens_count} tokens in {duration:.2f}s ({tokens_per_second:.2f} tok/s)", flush=True)
                return tokens_per_second
            else:
                print("[BENCHMARK] Error: No tokens generated.", flush=True)
                return None

        except Exception as e:
            print(f"[BENCHMARK] Request error: {e}", flush=True)
            return None

    finally:
        print("[BENCHMARK] Stopping server...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        kill_port_8000()

def main():
    print("==========================================================", flush=True)
    print("DRAFT BLOCK SIZE BENCHMARK (mlx-vlm speculative decoding)", flush=True)
    print("==========================================================", flush=True)

    results = {}
    for draft_block_size in [1, 2, 3]:
        tps = run_benchmark_for_config(draft_block_size)
        if tps is not None:
            results[draft_block_size] = tps

    if not results:
        print("\n[BENCHMARK] Error: Benchmarking failed for all configurations. Falling back to default (1).", flush=True)
        fastest_block = 1
    else:
        fastest_block = max(results, key=results.get)
        print("\n==========================================================", flush=True)
        print("BENCHMARK RESULTS SUMMARY:", flush=True)
        for k, v in results.items():
            print(f"  * Draft Block Size = {k}: {v:.2f} tokens/sec", flush=True)
        print(f"Selected fastest configuration: {fastest_block} draft block size", flush=True)
        print("==========================================================", flush=True)

    with open(BENCHMARK_JSON, "w") as f:
        json.dump({"best_draft_block_size": fastest_block}, f, indent=4)

    print(f"Saved benchmark configuration to {BENCHMARK_JSON}", flush=True)

if __name__ == "__main__":
    main()
