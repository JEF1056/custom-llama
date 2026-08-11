#!/usr/bin/env python3
"""Benchmark three OpenAI-compatible machines across prompt lengths."""
import json, time, hashlib
from pathlib import Path

ENDPOINTS = [
    ('localhost (llamacpp)', 'http://localhost:8080/v1/chat/completions', '/models/qwen3.6-35b'),
    ('ml-2 (mlx)',         'http://ml-2:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
    ('ml-3 (mlx)',         'http://ml-3:8080/v1/chat/completions', '/Users/jfan/.qwen/models/qwen36-mlx'),
]

TEMPERATURE = 0.1
MAX_TOKENS = 2048
SYSTEM_PROMPT = "You are a coding assistant. Read the code/context provided, then solve the task below. Be precise, write clean code, and explain your reasoning concisely."

TASKS = {
    "short": {
        "desc": "Short (4k) — refactoring task",
        "prompt": "Given the following Python function that parses CSV data, refactor it to use a class-based approach with proper error handling and type hints. Write the refactored version with a CsvParser class.",
        "target_tokens": 4000,
    },
    "medium": {
        "desc": "Medium (40k) — multi-file codebase review",
        "prompt": "I will provide a large Python codebase for a REST API. Please review it and identify bugs, then provide fixes.",
        "target_tokens": 40000,
    },
    "long": {
        "desc": "Long (100k) — full codebase understanding",
        "prompt": "I will provide a very large Python codebase for a web application. After reading all of it, answer specific questions about the architecture and fix a reported bug. Specifically: (1) Identify the main entry point and explain the request flow. (2) Find any potential race conditions or concurrency issues. (3) Suggest improvements to the error handling strategy.",
        "target_tokens": 100000,
    },
}


def generate_context(target_tokens, seed):
    """Generate deterministic context content of approximately target_tokens tokens.
    Each ~4 chars ≈ 1 token. Per-unit sizes: model_class=616(154tok), service=350(88tok),
    utility=120(30tok), async_handler=150(38tok).
    We allocate fractions of target_chars to each section."""
    h = hashlib.sha256(str(seed).encode()).hexdigest()
    lines = []
    lines.append(f"# Codebase context - ~{target_tokens//2} tokens (seed={seed})")
    lines.append("import os, sys, json, logging, dataclasses, typing, abc")
    lines.append("from pathlib import Path")
    lines.append("from typing import Optional, List, Dict, Any, Tuple, Union")
    lines.append("")

    # Per-token budget: model=154tok, service=144tok, util=60tok, async=44tok
    # Allocate 20% of target to each section (80% total, 20% for headers/config)
    model_limit = int(target_tokens * 0.20 / 154)
    service_limit = int(target_tokens * 0.20 / 144)
    util_limit = int(target_tokens * 0.20 / 60)
    async_limit = int(target_tokens * 0.20 / 44)
    for i in range(model_limit):
        lines.append(f"@dataclasses.dataclass")
        lines.append(f"class ModelConfig_{i}:")
        lines.append(f'    name: str = "model_{i}"')
        lines.append(f"    dims: int = 512")
        lines.append(f"    layers: int = {2 + (i % 8)}")
        lines.append(f"    heads: int = {4 + (i % 8)}")
        lines.append(f"    dropout: float = 0.1")
        lines.append(f'    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)')
        lines.append("")
        lines.append(f"    def validate(self) -> bool:")
        lines.append(f'        logging.debug(f"Validating model_{i}")')
        lines.append(f"        return self.dims > 0 and self.layers > 0 and self.heads > 0")
        lines.append("")
        lines.append(f"    def to_dict(self) -> Dict[str, Any]:")
        lines.append(f"        return dataclasses.asdict(self)")
        lines.append("")
        lines.append(f"    @classmethod")
        lines.append(f"    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig_{i}':")
        lines.append(f'        return cls(**{{k:v for k,v in d.items() if k in cls.__dataclass_fields__}})')
        lines.append("")

    # Generate service classes
    for i in range(service_limit):
        lines.append(f"class Service_{i}:")
        lines.append(f'    """Service layer component {i}."""')
        lines.append(f"    def __init__(self, config=None):")
        lines.append(f"        self.config = config")
        lines.append(f"        self._cache = {{}}")
        lines.append(f"        self._initialized = False")
        lines.append(f"        logging.info(f'Service_{i} initialized')")
        lines.append("")
        lines.append(f"    def initialize(self) -> bool:")
        lines.append(f"        if self._initialized:")
        lines.append(f"            return True")
        lines.append(f'        logging.info(f"Initializing Service_{i}")')
        lines.append(f"        self._initialized = True")
        lines.append(f"        return True")
        lines.append("")
        lines.append(f"    def process(self, data) -> dict:")
        lines.append(f'        """Process data through service {i}."""')
        lines.append(f"        if not self._initialized:")
        lines.append(f"            self.initialize()")
        lines.append(f"        return {{")
        lines.append(f'            "service": "Service_{i}",')
        lines.append(f'            "status": "processed",')
        lines.append(f'            "input_size": len(str(data)) if data else 0,')
        lines.append(f'            "output": self._transform(data)')
        lines.append(f"        }}")
        lines.append(f"        self._cache[id(data)] = True")
        lines.append(f"        return True")
        lines.append("")
        lines.append(f"    def _transform(self, data) -> str:")
        lines.append(f"        if isinstance(data, dict):")
        lines.append(f'            return {{k: str(v) for k, v in data.items()}}')
        lines.append(f"        return str(data)")
        lines.append("")
        lines.append(f"    def health_check(self) -> bool:")
        lines.append(f"        return self._initialized")
        lines.append("")

    # Generate utility functions
    for i in range(util_limit):
        lines.append(f"def utility_func_{i}(data, threshold=0.5):")
        lines.append(f'    """Utility function {i}."""')
        lines.append(f"    results = []")
        lines.append(f"    for item in data:")
        lines.append(f"        score = hash(str(item)) % 100 / 100.0")
        lines.append(f"        if score >= threshold:")
        lines.append(f"            results.append({{")
        lines.append(f'                "item": str(item),')
        lines.append(f'                "score": score,')
        lines.append(f'                "pass": True')
        lines.append(f"            }})")
        lines.append(f"    return results")
        lines.append("")
        lines.append("")

    # Generate config section
    lines.append("# === Configuration ===")
    lines.append('CONFIG_VERSION = "2.1.0"')
    lines.append("DEFAULT_TIMEOUT = 30.0")
    lines.append("MAX_RETRIES = 3")
    lines.append("")
    lines.append("class AppConfig:")
    lines.append('    """Global application configuration."""')
    lines.append("    def __init__(self):")
    lines.append("        self.services = {}")
    lines.append("        self.models = []")
    lines.append("        self.logging_level = logging.INFO")
    lines.append('        self.debug = os.environ.get("DEBUG", "false").lower() == "true"')
    lines.append("")
    lines.append("    def register_service(self, name, service):")
    lines.append("        self.services[name] = service")
    lines.append('        logging.info(f"Registered service: {name}")')
    lines.append("")
    lines.append("    def get_service(self, name):")
    lines.append("        return self.services.get(name)")
    lines.append("")
    lines.append("    def list_services(self):")
    lines.append("        return list(self.services.keys())")
    lines.append("")
    lines.append("")

    # Add async functions
    for i in range(async_limit):
        lines.append(f"async def async_handler_{i}(request):")
        lines.append(f'    """Async request handler {i}."""')
        lines.append(f"    try:")
        lines.append(f'        logging.debug(f"Handling request {i}")')
        lines.append(f"        await asyncio.sleep(0.001)")
        lines.append(f"        return {{")
        lines.append(f'            "handler": f"handler_{i}",')
        lines.append(f'            "status": "ok",')
        lines.append(f'            "request_id": request.get("id", "unknown")')
        lines.append(f"        }}")
        lines.append(f'    except Exception as e:')
        lines.append(f'        logging.error(f"Handler {i} error: {{e}}")')
        lines.append(f"        raise")
        lines.append("")
        lines.append("")

    # Pad with comments to reach target size
    while sum(len(l) for l in lines) * 1.3 < target_tokens * 4:
        idx = len(lines)
        lines.append(f"# Context block {idx}: additional processing pipeline configuration")
        lines.append(f"# This section handles input normalization for service layer.")

    return "\n".join(lines)


def count_tokens_approx(text):
    """Rough token count: ~4 chars per token."""
    return len(text) // 4


def run_benchmark():
    all_results = {}

    for size_name, task_info in TASKS.items():
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {task_info['desc']}")
        print(f"{'='*70}")

        target_tokens = task_info["target_tokens"]
        context = generate_context(target_tokens, seed=size_name)
        actual_tokens = count_tokens_approx(context)
        print(f"  Generated context: ~{actual_tokens:,} tokens ({len(context):,} chars)")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{task_info['prompt']}\n\n---\n\nHere is the codebase context:\n\n```\n{context}\n```"},
        ]

        for endpoint_name, url, model in ENDPOINTS:
            print(f"\n  -> {endpoint_name}...", end=" ", flush=True)
            try:
                import httpx
                client = httpx.Client(timeout=600)
                start = time.time()
                resp = client.post(
                    url,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "stream": False,
                    },
                )
                elapsed = time.time() - start
                resp.raise_for_status()
                data = resp.json()

                usage = data.get("usage", {})
                timings = data.get("timings", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                completion_text = data["choices"][0]["message"]["content"]

                predicted_per_sec = timings.get("predicted_per_second", 0)
                prompt_per_sec = timings.get("prompt_per_second", 0)
                actual_dps = predicted_per_sec if predicted_per_sec > 0 else (completion_tokens / elapsed if elapsed > 0 else 0)

                print(f"OK ({elapsed:.1f}s, {actual_dps:.1f} tok/s, {completion_tokens} tok)")
                all_results.setdefault(size_name, {})[endpoint_name] = {
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "decode_tok_per_sec": round(actual_dps, 1),
                    "prompt_tok_per_sec": round(prompt_per_sec, 1) if prompt_per_sec > 0 else None,
                    "response_len": len(completion_text),
                }
                client.close()

            except Exception as e:
                print(f"FAILED ({e})")
                all_results.setdefault(size_name, {})[endpoint_name] = {"error": str(e)}

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    for size_name in TASKS:
        print(f"\n  [{TASKS[size_name]['desc']}]")
        print(f"  {'Machine':<22} {'Time(s)':>8} {'P-tok/s':>10} {'D-tok/s':>10} {'Prompt':>10} {'Decode':>10}")
        print(f"  {'_'*22} {'_'*8} {'_'*10} {'_'*10} {'_'*10} {'_'*10}")
        for endpoint_name, result in all_results.get(size_name, {}).items():
            if "error" in result:
                print(f"  {endpoint_name:<22} {'ERROR':>8}")
            else:
                pps = result.get('prompt_tok_per_sec') or result['decode_tok_per_sec']
                print(f"  {endpoint_name:<22} {result['elapsed_s']:>8.1f} {pps:>10.1f} {result['decode_tok_per_sec']:>10.1f} {result['prompt_tokens']:>10,} {result['completion_tokens']:>10,}")

    # Save results
    results_path = Path("/tmp/benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_benchmark()
