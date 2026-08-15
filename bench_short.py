#!/usr/bin/env python3
"""Benchmark short prompt (4k) across all 3 machines."""
import json, time, httpx

ENDPOINTS = [
    ('localhost (llamacpp)', 'http://localhost:8080/v1/chat/completions', '/models/qwen3.6-35b'),
    ('ml-2 (mlx)',         'http://ml-2:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
    ('ml-3 (mlx)',         'http://ml-3:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
]

SYSTEM = "You are a coding assistant. Read the code/context provided, then solve the task below. Be precise, write clean code, and explain your reasoning concisely."

# Generate ~4k token context
ctx_lines = []
ctx_lines.append("# Codebase context - short prompt benchmark")
ctx_lines.append("import os, sys, json, logging, dataclasses, typing, abc")
ctx_lines.append("from pathlib import Path")
ctx_lines.append("from typing import Optional, List, Dict, Any, Tuple, Union")
ctx_lines.append("")
for i in range(5):
    ctx_lines.append(f"@dataclasses.dataclass")
    ctx_lines.append(f"class ModelConfig_{i}:")
    ctx_lines.append(f'    name: str = "model_{i}"')
    ctx_lines.append(f"    dims: int = 512")
    ctx_lines.append(f"    layers: int = {2 + (i % 8)}")
    ctx_lines.append(f"    heads: int = {4 + (i % 8)}")
    ctx_lines.append(f"    dropout: float = 0.1")
    ctx_lines.append(f'    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)')
    ctx_lines.append("")
    ctx_lines.append(f"    def validate(self) -> bool:")
    ctx_lines.append(f'        logging.debug(f"Validating model_{i}")')
    ctx_lines.append(f"        return self.dims > 0 and self.layers > 0 and self.heads > 0")
    ctx_lines.append("")
    ctx_lines.append(f"    def to_dict(self) -> Dict[str, Any]:")
    ctx_lines.append(f"        return dataclasses.asdict(self)")
    ctx_lines.append("")
    ctx_lines.append(f"    @classmethod")
    ctx_lines.append(f"    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig_{i}':")
    ctx_lines.append(f'        return cls(**{{k:v for k,v in d.items() if k in cls.__dataclass_fields__}})')
    ctx_lines.append("")
for i in range(5):
    ctx_lines.append(f"class Service_{i}:")
    ctx_lines.append(f'    """Service layer component {i}."""')
    ctx_lines.append(f"    def __init__(self, config=None):")
    ctx_lines.append(f"        self.config = config")
    ctx_lines.append(f"        self._cache = {{}}")
    ctx_lines.append(f"        self._initialized = False")
    ctx_lines.append(f"        logging.info(f'Service_{i} initialized')")
    ctx_lines.append("")
    ctx_lines.append(f"    def initialize(self) -> bool:")
    ctx_lines.append(f"        if self._initialized:")
    ctx_lines.append(f"            return True")
    ctx_lines.append(f'        logging.info(f"Initializing Service_{i}")')
    ctx_lines.append(f"        self._initialized = True")
    ctx_lines.append(f"        return True")
    ctx_lines.append("")
    ctx_lines.append(f"    def process(self, data) -> dict:")
    ctx_lines.append(f'        """Process data through service {i}."""')
    ctx_lines.append(f"        if not self._initialized:")
    ctx_lines.append(f"            self.initialize()")
    ctx_lines.append(f"        return {{")
    ctx_lines.append(f'            "service": "Service_{i}",')
    ctx_lines.append(f'            "status": "processed",')
    ctx_lines.append(f'            "input_size": len(str(data)) if data else 0,')
    ctx_lines.append(f'            "output": self._transform(data)')
    ctx_lines.append(f"        }}")
    ctx_lines.append(f"        self._cache[id(data)] = True")
    ctx_lines.append(f"        return True")
    ctx_lines.append("")
    ctx_lines.append(f"    def _transform(self, data) -> str:")
    ctx_lines.append(f"        if isinstance(data, dict):")
    ctx_lines.append(f'            return {{k: str(v) for k, v in data.items()}}')
    ctx_lines.append(f"        return str(data)")
    ctx_lines.append("")
    ctx_lines.append(f"    def health_check(self) -> bool:")
    ctx_lines.append(f"        return self._initialized")
    ctx_lines.append("")
for i in range(10):
    ctx_lines.append(f"def utility_func_{i}(data, threshold=0.5):")
    ctx_lines.append(f'    """Utility function {i}."""')
    ctx_lines.append(f"    results = []")
    ctx_lines.append(f"    for item in data:")
    ctx_lines.append(f"        score = hash(str(item)) % 100 / 100.0")
    ctx_lines.append(f"        if score >= threshold:")
    ctx_lines.append(f"            results.append({{")
    ctx_lines.append(f'                "item": str(item),')
    ctx_lines.append(f'                "score": score,')
    ctx_lines.append(f'                "pass": True')
    ctx_lines.append(f"            }})")
    ctx_lines.append(f"    return results")
    ctx_lines.append("")
    ctx_lines.append("")
ctx_lines.append("# === Configuration ===")
ctx_lines.append('CONFIG_VERSION = "2.1.0"')
ctx_lines.append("DEFAULT_TIMEOUT = 30.0")
ctx_lines.append("MAX_RETRIES = 3")
ctx_lines.append("")
ctx_lines.append("class AppConfig:")
ctx_lines.append('    """Global application configuration."""')
ctx_lines.append("    def __init__(self):")
ctx_lines.append("        self.services: Dict[str, Any] = {}")
ctx_lines.append("        self.models: List[Any] = []")
ctx_lines.append("        self.logging_level = logging.INFO")
ctx_lines.append('        self.debug = os.environ.get("DEBUG", "false").lower() == "true"')
ctx_lines.append("")
ctx_lines.append("    def register_service(self, name: str, service: Any) -> None:")
ctx_lines.append("        self.services[name] = service")
ctx_lines.append('        logging.info(f"Registered service: {name}")')
ctx_lines.append("")
ctx_lines.append("    def get_service(self, name: str) -> Optional[Any]:")
ctx_lines.append("        return self.services.get(name)")
ctx_lines.append("")
ctx_lines.append("    def list_services(self) -> List[str]:")
ctx_lines.append("        return list(self.services.keys())")
ctx_lines.append("")
ctx_lines.append("")
for i in range(5):
    ctx_lines.append(f"async def async_handler_{i}(request: Dict[str, Any]) -> Any:")
    ctx_lines.append(f'    """Async request handler {i}."""')
    ctx_lines.append(f"    try:")
    ctx_lines.append(f'        logging.debug(f"Handling request {i}")')
    ctx_lines.append(f"        await asyncio.sleep(0.001)")
    ctx_lines.append(f"        return {{")
    ctx_lines.append(f'            "handler": f"handler_{i}",')
    ctx_lines.append(f'            "status": "ok",')
    ctx_lines.append(f'            "request_id": request.get("id", "unknown")')
    ctx_lines.append(f"        }}")
    ctx_lines.append(f'    except Exception as e:')
    ctx_lines.append(f'        logging.error(f"Handler {i} error: {{e}}")')
    ctx_lines.append(f"        raise")
    ctx_lines.append("")
    ctx_lines.append("")

# Pad to ~4k tokens
while len('\n'.join(ctx_lines)) < 16000:
    ctx_lines.append("# Context block: additional processing pipeline configuration")

ctx = '\n'.join(ctx_lines)
actual_tokens = len(ctx) // 4
print(f"Context: ~{actual_tokens:,} tokens ({len(ctx):,} chars)")

msg = [
    {'role': 'system', 'content': SYSTEM},
    {'role': 'user', 'content': f'Refactor this CSV parser to a class-based approach with error handling and type hints.\n\n---\n\nHere is the codebase context:\n\n```\n{ctx}\n```'},
]

print()
print(f'{"Machine":<22} {"Time(s)":>8} {"P-tok/s":>10} {"D-tok/s":>10} {"Prompt":>10} {"Decode":>10}')
print(f'{"_"*22} {"_"*8} {"_"*10} {"_"*10} {"_"*10} {"_"*10}')

results = {}
for name, url, model in ENDPOINTS:
    try:
        c = httpx.Client(timeout=120)
        t0 = time.time()
        r = c.post(url, json={
            'model': model,
            'messages': msg,
            'temperature': 0.1,
            'max_tokens': 2048,
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
        print(f'{name:<22} {"ERROR":>8} {"":>10} {"":>10} {"":>10} {"":>10}')
        results[name] = {'error': str(e)}

# Save results
with open('/tmp/bench_short_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to /tmp/bench_short_results.json")
