#!/usr/bin/env python3
"""vLLM benchmark orchestrator — phased throughput testing.

Phases (most → least impactful):
  1. Speculative decoding sweep  (4 configs, ~55 min)
  2. KV cache dtype sweep        (4 configs, ~55 min)
  3. DRY on/off                  (per-request, ~15 min)
  4. Cross-validation of top 3   (5 runs each, ~40 min)

Results are flushed to JSONL after every individual run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from benchmark_scenarios import SCENARIOS, Scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
COMPOSE_FILES = [
    "-f", str(PROJECT_ROOT / "docker-compose.yml"),
    "-f", str(PROJECT_ROOT / "docker-compose.benchmark.yml"),
]
API_BASE = "http://localhost:8080"
HEALTH_URL = f"{API_BASE}/health"
CHAT_URL = f"{API_BASE}/v1/chat/completions"
METRICS_URL = f"{API_BASE}/metrics"

MAX_TOKENS = 1024
HEALTH_POLL_INTERVAL = 10  # seconds
HEALTH_TIMEOUT = 420  # 7 minutes — covers 5-min model load + compilation

# Default server-side config (held constant unless being tested)
DEFAULT_KV = "rotorquant_k4v2_nc"
DEFAULT_SPEC = ""  # disabled

DRY_ON_PARAMS = {
    "dry_multiplier": 0.4,
    "dry_base": 1.75,
    "dry_allowed_length": 128,
    "dry_penalty_last_n": 2048,
}

# Speculative decoding configs for Phase 1
# Sweeps: num_speculative_tokens (1,2,3), ngram settings, and combinations.
SPEC_CONFIGS: dict[str, str] = {
    # Baseline — no speculation
    "none": "",

    # MTP only — sweep token count
    "mtp_n1": json.dumps({"method": "mtp", "num_speculative_tokens": 1}),
    "mtp_n2": json.dumps({"method": "mtp", "num_speculative_tokens": 2}),
    "mtp_n3": json.dumps({"method": "mtp", "num_speculative_tokens": 3}),

    # Ngram only — sweep lookup window size
    "ngram_tight": json.dumps({
        "method": "ngram", "num_speculative_tokens": 2,
        "prompt_lookup_max": 6, "prompt_lookup_min": 3,
    }),
    "ngram_default": json.dumps({
        "method": "ngram", "num_speculative_tokens": 2,
        "prompt_lookup_max": 10, "prompt_lookup_min": 8,
    }),
    "ngram_wide": json.dumps({
        "method": "ngram", "num_speculative_tokens": 2,
        "prompt_lookup_max": 15, "prompt_lookup_min": 5,
    }),

    # Ngram-first + MTP — sweep token count (ngram settings = default)
    "ngram_mtp_n1": json.dumps({
        "method": "mtp", "num_speculative_tokens": 1,
        "ngram_first": True, "prompt_lookup_max": 10, "prompt_lookup_min": 8,
    }),
    "ngram_mtp_n2": json.dumps({
        "method": "mtp", "num_speculative_tokens": 2,
        "ngram_first": True, "prompt_lookup_max": 10, "prompt_lookup_min": 8,
    }),
    "ngram_mtp_n3": json.dumps({
        "method": "mtp", "num_speculative_tokens": 3,
        "ngram_first": True, "prompt_lookup_max": 10, "prompt_lookup_min": 8,
    }),
}

# KV cache dtype configs for Phase 2
KV_CONFIGS: dict[str, str] = {
    "rq_k4v2": "rotorquant_k4v2_nc",
    "rq_3bit": "rotorquant_3bit_nc",
    "tq_k4v2": "turboquant_k4v2_nc",
    "tq_3bit": "turboquant_3bit_nc",
}


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Terminal progress bar with ETA."""

    def __init__(self, total: int, phase_name: str):
        self.total = total
        self.phase_name = phase_name
        self.completed = 0
        self.start_time = time.monotonic()
        self._durations: list[float] = []
        self._cols = shutil.get_terminal_size((80, 24)).columns

    def update(self, label: str, last_tps: float | None = None, duration: float = 0.0):
        self.completed += 1
        if duration > 0:
            self._durations.append(duration)

        pct = self.completed / self.total if self.total else 1
        elapsed = time.monotonic() - self.start_time

        # ETA from running average
        if self._durations:
            avg = statistics.mean(self._durations)
            eta_s = avg * (self.total - self.completed)
            eta = _fmt_duration(eta_s)
        elif elapsed > 0 and self.completed > 0:
            eta_s = (elapsed / self.completed) * (self.total - self.completed)
            eta = _fmt_duration(eta_s)
        else:
            eta = "???"

        bar_width = min(30, self._cols - 60)
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        tps_str = f"{last_tps:.1f} tok/s" if last_tps is not None else ""

        line = (
            f"\r[{self.phase_name}] {bar} {self.completed}/{self.total} "
            f"({pct:4.0%}) | {label} | {tps_str} | ETA: {eta}"
        )
        # Truncate to terminal width, pad to clear previous line
        line = line[: self._cols].ljust(self._cols)
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self):
        elapsed = _fmt_duration(time.monotonic() - self.start_time)
        sys.stderr.write(f"\n  Phase complete in {elapsed}\n")
        sys.stderr.flush()


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Docker lifecycle
# ---------------------------------------------------------------------------

def restart_server(
    kv_cache_dtype: str,
    speculative_config: str,
    dry_config: str = "",
) -> None:
    """Force-recreate vllm-server with the given env overrides."""
    env = os.environ.copy()
    env["LLM_KV_CACHE_DTYPE"] = kv_cache_dtype
    env["LLM_SPECULATIVE_CONFIG"] = speculative_config
    env["LLM_DRY_CONFIG"] = dry_config
    # Don't set LLM_DISABLE_LOG_REQUESTS — let .env default control it

    cmd = [
        "docker", "compose", *COMPOSE_FILES,
        "up", "-d", "--force-recreate", "vllm-server",
    ]
    _log(f"Restarting server: kv={kv_cache_dtype} spec={speculative_config or '(none)'}")
    _log(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, env=env, cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    # Always show docker compose output so restarts are visible
    if result.stdout.strip():
        _log(f"  stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        _log(f"  stderr: {result.stderr.strip()}")
    if result.returncode != 0:
        raise RuntimeError(f"docker compose restart failed (rc={result.returncode})")


def wait_for_health(timeout: int = HEALTH_TIMEOUT) -> float:
    """Poll /health until 200. Returns seconds waited."""
    _log(f"Waiting for health (timeout {timeout}s)...")
    start = time.monotonic()
    deadline = start + timeout
    with httpx.Client(timeout=10) as client:
        while time.monotonic() < deadline:
            try:
                r = client.get(HEALTH_URL)
                if r.status_code == 200:
                    waited = time.monotonic() - start
                    _log(f"Server healthy after {waited:.0f}s")
                    return waited
            except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout,
                    httpx.RemoteProtocolError, httpx.WriteError, ConnectionError):
                pass
            time.sleep(HEALTH_POLL_INTERVAL)
    raise TimeoutError(f"Server not healthy after {timeout}s")


def stop_server() -> None:
    """Stop vllm-server (but not other services)."""
    subprocess.run(
        ["docker", "compose", *COMPOSE_FILES, "stop", "vllm-server"],
        cwd=PROJECT_ROOT, capture_output=True,
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def read_api_key() -> str:
    """Read LLM_API_KEY from .env if it exists."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("LLM_API_KEY="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            return val
    return ""


def scrape_spec_metrics() -> dict[str, float]:
    """Scrape speculative decoding metrics from vLLM's Prometheus endpoint.

    Returns a dict with acceptance_rate, accepted_tokens, drafted_tokens, etc.
    Returns empty dict if metrics unavailable or spec decoding is off.
    """
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(METRICS_URL)
            if r.status_code != 200:
                return {}
    except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout, ConnectionError):
        return {}

    metrics: dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        # Parse Prometheus text format: metric_name{labels} value
        for name, key in [
            ("vllm:spec_decode_draft_acceptance_rate", "spec_acceptance_rate"),
            ("vllm:spec_decode_efficiency", "spec_efficiency"),
            ("vllm:spec_decode_num_accepted_tokens_total", "spec_accepted_tokens"),
            ("vllm:spec_decode_num_draft_tokens_total", "spec_drafted_tokens"),
            ("vllm:spec_decode_num_emitted_tokens_total", "spec_emitted_tokens"),
            # Also try underscore variants (metric naming varies by vLLM version)
            ("vllm_spec_decode_draft_acceptance_rate", "spec_acceptance_rate"),
            ("vllm_spec_decode_efficiency", "spec_efficiency"),
            ("vllm_spec_decode_num_accepted_tokens_total", "spec_accepted_tokens"),
            ("vllm_spec_decode_num_draft_tokens_total", "spec_drafted_tokens"),
            ("vllm_spec_decode_num_emitted_tokens_total", "spec_emitted_tokens"),
        ]:
            if line.startswith(name):
                try:
                    # Handle both "metric value" and "metric{labels} value"
                    val_str = line.split()[-1]
                    val = float(val_str)
                    if key not in metrics:  # first match wins
                        metrics[key] = val
                except (ValueError, IndexError):
                    pass

    return metrics


def run_scenario(
    scenario: Scenario,
    dry: bool = False,
    api_key: str = "",
) -> dict[str, Any]:
    """Execute one benchmark run via streaming chat completion. Returns metrics."""
    messages = [
        {"role": "system", "content": scenario.system},
        {"role": "user", "content": scenario.user},
    ]

    body: dict[str, Any] = {
        "model": "qwen3.6-27b",
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # Disable thinking for consistent throughput measurement
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
    }

    if dry:
        extra_body.update(DRY_ON_PARAMS)

    body.update(extra_body)

    if scenario.tools:
        body["tools"] = scenario.tools
        body["tool_choice"] = "auto"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t_start = time.monotonic()
    t_first_token: float | None = None
    t_last_token = t_start
    prompt_tokens = 0
    completion_tokens = 0
    chunks_received = 0

    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        with client.stream("POST", CHAT_URL, json=body, headers=headers) as response:
            if response.status_code != 200:
                error_body = response.read().decode(errors="replace")
                raise RuntimeError(
                    f"API returned {response.status_code}: {error_body[:500]}"
                )
            for raw_line in response.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage chunk (sent last with stream_options.include_usage)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                # Content chunk
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or delta.get("tool_calls")
                    if content:
                        now = time.monotonic()
                        if t_first_token is None:
                            t_first_token = now
                        t_last_token = now
                        chunks_received += 1

    total_time = t_last_token - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    decode_time = (t_last_token - t_first_token) if t_first_token else 0.001

    result = {
        "ttft_s": round(ttft, 3),
        "decode_tps": round(completion_tokens / decode_time, 2) if decode_time > 0 else 0,
        "overall_tps": round(completion_tokens / total_time, 2) if total_time > 0 else 0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": round(total_time, 3),
        "chunks": chunks_received,
    }

    # Scrape speculative decoding metrics from Prometheus endpoint
    spec_metrics = scrape_spec_metrics()
    if spec_metrics:
        result["spec_metrics"] = spec_metrics

    return result


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def flush_result(result: dict[str, Any], path: Path) -> None:
    """Append one result to JSONL immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load all results from JSONL."""
    if not path.exists():
        return []
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def result_key(r: dict) -> tuple:
    """Unique key for deduplication / resume."""
    sc = r.get("server_config", {})
    rc = r.get("request_config", {})
    return (
        r.get("phase"),
        sc.get("kv_cache_dtype"),
        sc.get("speculative_config"),
        rc.get("dry"),
        r.get("scenario"),
        r.get("run"),
    )


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_config_sweep(
    *,
    phase: int,
    phase_name: str,
    configs: list[dict[str, str]],
    config_label_key: str,
    fixed_kv: str,
    fixed_spec: str,
    dry: bool,
    runs_per_scenario: int,
    results_path: Path,
    completed: set[tuple],
    api_key: str,
) -> list[dict[str, Any]]:
    """Generic sweep: restart per config, run all scenarios × runs."""
    total = len(configs) * len(SCENARIOS) * runs_per_scenario
    tracker = ProgressTracker(total, f"Phase {phase}: {phase_name}")
    all_results: list[dict[str, Any]] = []

    for cfg in configs:
        kv = cfg.get("kv_cache_dtype", fixed_kv)
        spec = cfg.get("speculative_config", fixed_spec)
        label = cfg[config_label_key]

        restart_server(kv, spec)
        restart_time_start = time.monotonic()
        wait_for_health()
        restart_duration = time.monotonic() - restart_time_start

        # Warmup — discard
        try:
            run_scenario(SCENARIOS[0], dry=False, api_key=api_key)
        except Exception as e:
            _log(f"Warmup failed ({e}), continuing anyway")

        for scenario in SCENARIOS:
            for run_num in range(1, runs_per_scenario + 1):
                key = (phase, kv, spec, dry, scenario.name, run_num)
                if key in completed:
                    tracker.update(f"{label} x {scenario.name} run {run_num} (cached)")
                    continue

                run_start = time.monotonic()
                try:
                    metrics = run_scenario(scenario, dry=dry, api_key=api_key)
                    status = "ok"
                    error = None
                except Exception as e:
                    _log(f"\nERROR: {label} x {scenario.name} run {run_num}: {e}")
                    metrics = {}
                    status = "error"
                    error = str(e)

                result = {
                    "phase": phase,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_cache_dtype": kv,
                        "speculative_config": spec,
                        "config_label": label,
                    },
                    "request_config": {"dry": dry},
                    "scenario": scenario.name,
                    "run": run_num,
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                    "restart_duration_s": round(restart_duration, 1),
                }
                flush_result(result, results_path)
                all_results.append(result)

                run_duration = time.monotonic() - run_start
                tps = metrics.get("overall_tps") if metrics else None
                tracker.update(
                    f"{label} x {scenario.name} run {run_num}/{runs_per_scenario}",
                    last_tps=tps,
                    duration=run_duration,
                )

    tracker.finish()
    return all_results


def run_phase_1(
    runs: int, results_path: Path, completed: set[tuple], api_key: str,
) -> str:
    """Phase 1: Speculative decoding sweep. Returns winner spec config."""
    _log("\n=== Phase 1: Speculative Decoding ===")
    configs = [
        {"speculative_config": v, "config_label": k}
        for k, v in SPEC_CONFIGS.items()
    ]
    results = _run_config_sweep(
        phase=1,
        phase_name="Speculative",
        configs=configs,
        config_label_key="config_label",
        fixed_kv=DEFAULT_KV,
        fixed_spec="",  # overridden per config
        dry=False,
        runs_per_scenario=runs,
        results_path=results_path,
        completed=completed,
        api_key=api_key,
    )
    return _pick_winner(results, "speculative_config")


def run_phase_2(
    runs: int, results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: str,
) -> str:
    """Phase 2: KV cache dtype sweep. Returns winner KV dtype."""
    _log("\n=== Phase 2: KV Cache Dtype ===")
    configs = [
        {"kv_cache_dtype": v, "config_label": k}
        for k, v in KV_CONFIGS.items()
    ]
    results = _run_config_sweep(
        phase=2,
        phase_name="KV Cache",
        configs=configs,
        config_label_key="config_label",
        fixed_kv="",  # overridden per config
        fixed_spec=spec_winner,
        dry=False,
        runs_per_scenario=runs,
        results_path=results_path,
        completed=completed,
        api_key=api_key,
    )
    return _pick_winner(results, "kv_cache_dtype")


def run_phase_3(
    runs: int, results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: str, kv_winner: str,
) -> None:
    """Phase 3: DRY on vs off (per-request, no restart)."""
    _log("\n=== Phase 3: DRY Sampling ===")

    # Validate per-request DRY works
    _log("Validating per-request DRY...")
    dry_works = _validate_dry_per_request(api_key)

    if dry_works:
        # No restart needed — server already running from Phase 2 tail
        # (or restart once to ensure correct kv + spec)
        restart_server(kv_winner, spec_winner)
        wait_for_health()

        total = 2 * len(SCENARIOS) * runs
        tracker = ProgressTracker(total, "Phase 3: DRY")

        for dry in [False, True]:
            dry_label = "dry_on" if dry else "dry_off"
            for scenario in SCENARIOS:
                for run_num in range(1, runs + 1):
                    key = (3, kv_winner, spec_winner, dry, scenario.name, run_num)
                    if key in completed:
                        tracker.update(f"{dry_label} x {scenario.name} run {run_num} (cached)")
                        continue

                    run_start = time.monotonic()
                    try:
                        metrics = run_scenario(scenario, dry=dry, api_key=api_key)
                        status = "ok"
                        error = None
                    except Exception as e:
                        _log(f"\nERROR: {dry_label} x {scenario.name} run {run_num}: {e}")
                        metrics = {}
                        status = "error"
                        error = str(e)

                    result = {
                        "phase": 3,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "server_config": {
                            "kv_cache_dtype": kv_winner,
                            "speculative_config": spec_winner,
                            "config_label": f"{dry_label}",
                        },
                        "request_config": {"dry": dry},
                        "scenario": scenario.name,
                        "run": run_num,
                        "metrics": metrics,
                        "status": status,
                        "error": error,
                    }
                    flush_result(result, results_path)

                    run_duration = time.monotonic() - run_start
                    tps = metrics.get("overall_tps") if metrics else None
                    tracker.update(
                        f"{dry_label} x {scenario.name} run {run_num}/{runs}",
                        last_tps=tps,
                        duration=run_duration,
                    )

        tracker.finish()
    else:
        _log("Per-request DRY not supported — falling back to server-level restart")
        for dry in [False, True]:
            dry_config = json.dumps(DRY_ON_PARAMS) if dry else ""
            restart_server(kv_winner, spec_winner, dry_config=dry_config)
            wait_for_health()
            run_scenario(SCENARIOS[0], dry=False, api_key=api_key)  # warmup

            for scenario in SCENARIOS:
                for run_num in range(1, runs + 1):
                    key = (3, kv_winner, spec_winner, dry, scenario.name, run_num)
                    if key in completed:
                        continue
                    try:
                        metrics = run_scenario(scenario, dry=False, api_key=api_key)
                        status = "ok"
                        error = None
                    except Exception as e:
                        metrics = {}
                        status = "error"
                        error = str(e)
                    result = {
                        "phase": 3,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "server_config": {
                            "kv_cache_dtype": kv_winner,
                            "speculative_config": spec_winner,
                            "config_label": "dry_on" if dry else "dry_off",
                        },
                        "request_config": {"dry": dry},
                        "scenario": scenario.name,
                        "run": run_num,
                        "metrics": metrics,
                        "status": status,
                        "error": error,
                    }
                    flush_result(result, results_path)


def run_phase_4(
    results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: str, kv_winner: str,
) -> None:
    """Phase 4: Cross-validate top 3 full configs with 5 runs."""
    _log("\n=== Phase 4: Cross-validation ===")

    all_results = load_results(results_path)
    if not all_results:
        _log("No results to cross-validate. Skipping Phase 4.")
        return

    # Find top 3 distinct (kv, spec, dry) combos by mean overall_tps
    combo_tps: dict[tuple, list[float]] = {}
    for r in all_results:
        if r.get("status") != "ok":
            continue
        sc = r["server_config"]
        rc = r["request_config"]
        combo = (sc["kv_cache_dtype"], sc["speculative_config"], rc["dry"])
        tps = r["metrics"].get("overall_tps", 0)
        combo_tps.setdefault(combo, []).append(tps)

    ranked = sorted(
        combo_tps.items(),
        key=lambda x: statistics.mean(x[1]),
        reverse=True,
    )
    top3 = ranked[:3]
    _log(f"Top 3 configs for cross-validation:")
    for (kv, spec, dry), tps_list in top3:
        _log(f"  kv={kv} spec={spec or '(none)'} dry={dry} → {statistics.mean(tps_list):.1f} tok/s")

    crossval_runs = 5
    total = len(top3) * len(SCENARIOS) * crossval_runs
    tracker = ProgressTracker(total, "Phase 4: Cross-val")

    last_kv, last_spec = None, None
    for (kv, spec, dry), _ in top3:
        if kv != last_kv or spec != last_spec:
            restart_server(kv, spec)
            wait_for_health()
            run_scenario(SCENARIOS[0], dry=False, api_key=api_key)  # warmup
            last_kv, last_spec = kv, spec

        label = f"kv={kv[:6]} spec={_spec_label(spec)} dry={dry}"
        for scenario in SCENARIOS:
            for run_num in range(1, crossval_runs + 1):
                key = (4, kv, spec, dry, scenario.name, run_num)
                if key in completed:
                    tracker.update(f"{label} x {scenario.name} run {run_num} (cached)")
                    continue

                run_start = time.monotonic()
                try:
                    metrics = run_scenario(scenario, dry=dry, api_key=api_key)
                    status = "ok"
                    error = None
                except Exception as e:
                    metrics = {}
                    status = "error"
                    error = str(e)

                result = {
                    "phase": 4,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_cache_dtype": kv,
                        "speculative_config": spec,
                        "config_label": label,
                    },
                    "request_config": {"dry": dry},
                    "scenario": scenario.name,
                    "run": run_num,
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                }
                flush_result(result, results_path)

                run_duration = time.monotonic() - run_start
                tps = metrics.get("overall_tps") if metrics else None
                tracker.update(
                    f"{label} x {scenario.name} run {run_num}/{crossval_runs}",
                    last_tps=tps,
                    duration=run_duration,
                )

    tracker.finish()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_winner(results: list[dict], config_key: str) -> str:
    """Return the config value with the highest mean overall_tps."""
    by_config: dict[str, list[float]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        val = r["server_config"][config_key]
        tps = r["metrics"].get("overall_tps", 0)
        by_config.setdefault(val, []).append(tps)

    if not by_config:
        _log(f"WARNING: no successful results to pick winner for {config_key}")
        return DEFAULT_KV if config_key == "kv_cache_dtype" else DEFAULT_SPEC

    winner = max(by_config, key=lambda k: statistics.mean(by_config[k]))
    mean_tps = statistics.mean(by_config[winner])
    _log(f"Winner for {config_key}: {winner or '(none)'} ({mean_tps:.1f} tok/s mean)")

    # Log all contenders
    for val, tps_list in sorted(by_config.items(), key=lambda x: -statistics.mean(x[1])):
        m = statistics.mean(tps_list)
        s = statistics.stdev(tps_list) if len(tps_list) > 1 else 0
        _log(f"  {val or '(none)':30s}  {m:.1f} +/- {s:.1f} tok/s")

    return winner


def _validate_dry_per_request(api_key: str) -> bool:
    """Check if DRY params work via extra_body without server-level --dry-config."""
    scenario = SCENARIOS[0]
    try:
        r1 = run_scenario(scenario, dry=False, api_key=api_key)
        r2 = run_scenario(scenario, dry=True, api_key=api_key)
        # If both succeed, per-request DRY is accepted
        _log(f"DRY validation: off={r1.get('overall_tps', 0):.1f}, on={r2.get('overall_tps', 0):.1f} tok/s")
        return True
    except RuntimeError as e:
        if "400" in str(e) or "422" in str(e):
            _log(f"Per-request DRY rejected by server: {e}")
            return False
        raise


def _spec_label(spec: str) -> str:
    if not spec:
        return "none"
    try:
        d = json.loads(spec)
        method = d.get("method", "?")
        if d.get("ngram_first"):
            return "ngram+mtp"
        return method
    except json.JSONDecodeError:
        return spec[:15]


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    sys.stderr.write(f"\n[{ts}] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark orchestrator")
    parser.add_argument("--runs", type=int, default=3, help="Runs per scenario (default: 3)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run only this phase")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report from JSONL")
    parser.add_argument("--results-file", type=str, help="Path to existing JSONL (for resume/report)")
    args = parser.parse_args()

    # Determine results file path
    if args.results_file:
        results_path = Path(args.results_file)
    elif args.resume or args.report_only:
        # Find most recent JSONL in results dir
        existing = sorted(RESULTS_DIR.glob("*_runs.jsonl"))
        if not existing:
            _log("ERROR: no existing results found to resume/report from")
            sys.exit(1)
        results_path = existing[-1]
        _log(f"Using existing results: {results_path}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = RESULTS_DIR / f"{ts}_runs.jsonl"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        from benchmark_report import generate_report
        report_path = results_path.with_name(results_path.stem.replace("_runs", "_report") + ".md")
        generate_report(results_path, report_path)
        _log(f"Report written to {report_path}")
        return

    # Load completed results for resume
    completed: set[tuple] = set()
    if args.resume and results_path.exists():
        for r in load_results(results_path):
            completed.add(result_key(r))
        _log(f"Resuming: {len(completed)} runs already completed")

    api_key = read_api_key()
    _log(f"Results file: {results_path}")
    _log(f"Runs per scenario: {args.runs}")
    _log(f"API key: {'(set)' if api_key else '(none)'}")

    bench_start = time.monotonic()

    # Phase execution
    spec_winner = DEFAULT_SPEC
    kv_winner = DEFAULT_KV

    if args.phase is None or args.phase == 1:
        spec_winner = run_phase_1(args.runs, results_path, completed, api_key)
    elif args.phase and args.phase > 1:
        # Need to determine Phase 1 winner from existing results
        existing = [r for r in load_results(results_path) if r.get("phase") == 1]
        if existing:
            spec_winner = _pick_winner(existing, "speculative_config")

    if args.phase is None or args.phase == 2:
        kv_winner = run_phase_2(args.runs, results_path, completed, api_key, spec_winner)
    elif args.phase and args.phase > 2:
        existing = [r for r in load_results(results_path) if r.get("phase") == 2]
        if existing:
            kv_winner = _pick_winner(existing, "kv_cache_dtype")

    if args.phase is None or args.phase == 3:
        run_phase_3(args.runs, results_path, completed, api_key, spec_winner, kv_winner)

    if args.phase is None or args.phase == 4:
        run_phase_4(results_path, completed, api_key, spec_winner, kv_winner)

    total_time = time.monotonic() - bench_start
    _log(f"\nBenchmark complete in {_fmt_duration(total_time)}")
    _log(f"Results: {results_path}")

    # Auto-generate report
    try:
        from benchmark_report import generate_report
        report_path = results_path.with_name(results_path.stem.replace("_runs", "_report") + ".md")
        generate_report(results_path, report_path)
        _log(f"Report: {report_path}")
    except Exception as e:
        _log(f"Report generation failed: {e}")
        _log("Run manually: python scripts/benchmark.py --report-only")


if __name__ == "__main__":
    main()
