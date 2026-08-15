import urllib.request, json, time

prompt = "The quick brown fox jumps over the lazy dog. " * 3000

# Tokenize first
data = json.dumps({
    "model": "/Users/jfan/.qwen/models/qwen36-mlx",
    "input": [{"role": "user", "content": prompt}]
}).encode()
req = urllib.request.Request("http://localhost:8080/v1/responses/input_tokens", data=data, headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())
print(f"Token count: {resp['input_tokens']}")

# Now benchmark
payload = json.dumps({
    "model": "/Users/jfan/.qwen/models/qwen36-mlx",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 100,
    "temperature": 0.6,
    "stream": False
}).encode()
req2 = urllib.request.Request("http://localhost:8080/v1/chat/completions", data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer sk-noauth"})
start = time.time()
resp2 = json.loads(urllib.request.urlopen(req2, timeout=300).read())
elapsed = time.time() - start

timings = resp2.get("timings", {})
usage = resp2.get("usage", {})
print(f"Wall time: {elapsed:.1f}s")
print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
print(f"Completion tokens: {usage.get('completion_tokens', 0)}")
print(f"Prefill tok/s: {timings.get('prompt_per_second', 0):.1f}")
print(f"Decode tok/s: {timings.get('predicted_per_second', 0):.1f}")
print(f"Peak memory: {timings.get('peak_memory', 0):.1f} GB")
print(f"TTFT: {timings.get('prompt_ms', 0)/1000:.1f}s")
