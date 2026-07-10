import sys
import os
import time
import subprocess
import json
import urllib.request
import urllib.error

LOCAL_MODEL_PATH = "/Users/jfan/Documents/Qwen3.6-35B-A3B-MLX-4bit"
LAUNCHER_SCRIPT = "/Users/jfan/Documents/run_vllm_mlx.py"
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
        "model": "Qwen/Qwen3.6-35B-A3B",
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

def run_benchmark_for_config(num_draft_tokens):
    print(f"\n[BENCHMARK] Starting server with --mtp-num-draft-tokens {num_draft_tokens}...", flush=True)
    kill_port_8000()
    time.sleep(2)
    
    # Launch the server in the background
    cmd = [
        "python3", LAUNCHER_SCRIPT, "serve", LOCAL_MODEL_PATH,
        "--served-model-name", "Qwen/Qwen3.6-35B-A3B",
        "--enable-mtp",
        "--mtp-num-draft-tokens", str(num_draft_tokens),
        "--continuous-batching",
        "--use-paged-cache",
        "--enable-prefix-cache",
        "--kv-cache-quantization",
        "--kv-cache-quantization-bits", "4",
        "--cache-memory-percent", "10",
        "--prefill-batch-size", "1",
        "--completion-batch-size", "32",
        "--prefill-step-size", "1024",
        "--mllm-prefill-step-size", "1024",
        "--max-num-seqs", "16",
        "--default-thinking-token-budget", "2048",
        "--timeout", "3600",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen",
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        if not wait_for_server():
            print(f"[BENCHMARK] Error: Server failed to start for draft-tokens={num_draft_tokens}", flush=True)
            return None
            
        print(f"[BENCHMARK] Server is ready. Sending coding task request...", flush=True)
        
        # Define coding task payload
        payload = {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "messages": [
                {"role": "user", "content": "Write a highly optimized Python function to compute the longest common subsequence of two strings."}
            ],
            "max_tokens": 150,
            "temperature": 0.0,  # Greedy for maximum consistency
            "stream": True
        }
        
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        start_time = None
        first_token_time = None
        tokens_count = 0
        
        try:
            start_time = time.time()
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
                            
                            # Support both regular content and reasoning (thinking) content
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
    print("MTP CONFIGURATION BENCHMARK", flush=True)
    print("==========================================================", flush=True)
    
    results = {}
    for tokens in [1, 2, 3]:
        tps = run_benchmark_for_config(tokens)
        if tps is not None:
            results[tokens] = tps
            
    if not results:
        print("\n[BENCHMARK] Error: Benchmarking failed for all configurations. Falling back to default (1 token).", flush=True)
        fastest_tokens = 1
    else:
        fastest_tokens = max(results, key=results.get)
        print("\n==========================================================", flush=True)
        print("BENCHMARK RESULTS SUMMARY:", flush=True)
        for k, v in results.items():
            print(f"  * MTP Draft Tokens = {k}: {v:.2f} tokens/sec", flush=True)
        print(f"Selected fastest configuration: {fastest_tokens} draft tokens", flush=True)
        print("==========================================================", flush=True)
        
    with open(BENCHMARK_JSON, "w") as f:
        json.dump({"fastest_mtp_num_draft_tokens": fastest_tokens}, f, indent=4)
        
    print(f"Saved benchmark configuration to {BENCHMARK_JSON}", flush=True)

if __name__ == "__main__":
    main()
