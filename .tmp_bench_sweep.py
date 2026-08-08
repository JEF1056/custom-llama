import urllib.request, json, time

for tokens_mult in [1, 3, 5, 8, 10]:
    target_tokens = 10000 * tokens_mult
    prompt = "The quick brown fox jumps over the lazy dog. " * (target_tokens // 13)
    
    # Tokenize first
    data = json.dumps({
        "model": "/Users/jfan/.qwen/models/qwen36-mlx",
        "input": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request("http://localhost:8080/v1/responses/input_tokens", data=data, headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    actual_tokens = resp['input_tokens']
    
    # Benchmark
    payload = json.dumps({
        "model": "/Users/jfan/.qwen/models/qwen36-mlx",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.6,
        "stream": False
    }).encode()
    req2 = urllib.request.Request("http://localhost:8080/v1/chat/completions", data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer sk-noauth"})
    start = time.time()
    try:
        resp2 = json.loads(urllib.request.urlopen(req2, timeout=600).read())
        elapsed = time.time() - start
        timings = resp2.get("timings", {})
        usage = resp2.get("usage", {})
        print(f"Target: {target_tokens}K | Actual: {actual_tokens}K | Prefill: {timings.get('prompt_per_second', 0):.0f} tok/s | Decode: {timings.get('predicted_per_second', 0):.1f} tok/s | TTFT: {timings.get('prompt_ms', 0)/1000:.1f}s | Peak mem: {timings.get('peak_memory', 0):.0f}GB | Wall: {elapsed:.1f}s")
    except Exception as e:
        print(f"Target: {target_tokens}K | Actual: {actual_tokens}K | ERROR: {e}")
