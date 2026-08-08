import urllib.request, json, os

# Get current metrics
req = urllib.request.Request('http://localhost:8080/metrics')
metrics = json.loads(urllib.request.urlopen(req).read())

# Check APC stats
recent = metrics.get('recent', [])
apc_stats = recent[-1] if recent else {}
print("Latest request APC info:")
print(f"  apc_enabled: {apc_stats.get('apc_enabled', 'N/A')}")
print(f"  cached_tokens: {apc_stats.get('cached_tokens', 0)}")

# Check APC cache stats endpoint
req2 = urllib.request.Request('http://localhost:8080/v1/cache/stats')
stats = json.loads(urllib.request.urlopen(req2).read())
print("\nAPC Stats:")
print(f"  enabled: {stats.get('enabled', False)}")
print(f"  num_blocks: {stats.get('num_blocks', 0)}")
print(f"  pool_used: {stats.get('pool_used', 0)}")
print(f"  stores: {stats.get('stores', 0)}")
print(f"  exact_stores: {stats.get('exact_stores', 0)}")
print(f"  hits: {stats.get('lookups_hit', 0)}")
print(f"  misses: {stats.get('lookups_miss', 0)}")
print(f"  disk_bytes: {stats.get('disk_bytes', 0)}")
print(f"  exact_hits: {stats.get('exact_hits', 0)}")

# Check if disk tier has any cached data
cache_dir = '/Users/jfan/.cache/mlx-vlm/caching'
if os.path.exists(cache_dir):
    print(f"\nDisk cache contents:")
    for root, dirs, files in os.walk(cache_dir):
        level = root.replace(cache_dir, '').count(os.sep)
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = '  ' * (level + 1)
        for file in files[:10]:
            print(f'{subindent}{file}')
