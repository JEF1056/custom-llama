import urllib.request, json, time

# Force reset to check exact stores after reset
req = urllib.request.Request('http://localhost:8080/v1/cache/reset', data=b'', method='POST')
resp = json.loads(urllib.request.urlopen(req).read())
print('Cache reset:', resp)

# Now run a 60K token request
prompt = 'The quick brown fox jumps over the lazy dog. ' * 4600
data = json.dumps({
    'model': '/Users/jfan/.qwen/models/qwen36-mlx',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 50,
    'stream': False
}).encode()
req2 = urllib.request.Request('http://localhost:8080/v1/chat/completions', data=data, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer sk-noauth'})
start = time.perf_counter()
resp2 = json.loads(urllib.request.urlopen(req2, timeout=300).read())
elapsed = time.perf_counter() - start
timings = resp2.get('timings', {})
pn = timings.get('prompt_n', 0)
cn = timings.get('cache_n', 0)
pps = timings.get('prompt_per_second', 0)
dps = timings.get('predicted_per_second', 0)
print(f'Prompt tokens: {pn}')
print(f'Cache tokens: {cn}')
print(f'Prefill tok/s: {pps:.1f}')
print(f'Decode tok/s: {dps:.1f}')
print(f'Wall time: {elapsed:.1f}s')

# Check stats
req3 = urllib.request.Request('http://localhost:8080/v1/cache/stats')
stats = json.loads(urllib.request.urlopen(req3).read())
stores = stats.get('stores', 0)
exact_stores = stats.get('exact_stores', 0)
hits = stats.get('lookups_hit', 0)
pool_used = stats.get('pool_used', 0)
print(f'APC stores: {stores}')
print(f'APC exact_stores: {exact_stores}')
print(f'APC hits: {hits}')
print(f'APC pool_used: {pool_used}')
