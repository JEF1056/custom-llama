#!/usr/bin/env python3
"""llama.cpp benchmark orchestrator — phased throughput testing.

Phases (most -> least impactful):
  1. Speculative decoding sweep  (spec-type variants)
  2. KV cache dtype sweep        (turbo K/V bit combos)
  3. DRY on/off                  (sampling penalty impact)
  4. Cross-validation of top 3   (5 runs each)
  5. Stress tests                (long context, parallel slots, burst)

Results are flushed to JSONL after every individual run.

Usage:
  # Full phased benchmark (restarts server per config):
  python scripts/benchmark.py --runs 3

  # Benchmark against already-running server (no restarts):
  python scripts/benchmark.py --live --runs 5

  # Live mode with stress tests:
  python scripts/benchmark.py --live --runs 3 --stress

  # Resume an interrupted run:
  python scripts/benchmark.py --resume

  # Regenerate report from existing results:
  python scripts/benchmark.py --report-only
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
COMPOSE_FILE = str(PROJECT_ROOT / "docker-compose.yml")
API_BASE = "http://localhost:8080"
HEALTH_URL = f"{API_BASE}/health"
CHAT_URL = f"{API_BASE}/v1/chat/completions"

MAX_TOKENS = 1024
HEALTH_POLL_INTERVAL = 5   # seconds
HEALTH_TIMEOUT = 300       # 5 minutes — covers model load from disk

# llama.cpp models.ini values held constant unless being tested
DEFAULT_KV_K = "turbo4"
DEFAULT_KV_V = "turbo2"
DEFAULT_SPEC = "ngram-mod,draft-mtp"

# DRY sampling parameters (llama.cpp naming)
DRY_ON_PARAMS = {
    "dry_multiplier": 0.4,
    "dry_base": 1.75,
    "dry_allowed_length": 128,
    "dry_penalty_last_n": 2048,
}

# Speculative decoding configs for Phase 1
# llama.cpp spec-type format: comma-separated strategies, evaluated left-to-right
SPEC_CONFIGS: dict[str, dict[str, str]] = {
    # Baseline — no speculation
    "none": {
        "LLAMA_SPEC_TYPE": "",
        "LLAMA_SPEC_DRAFT_N_MAX": "0",
    },

    # MTP only (draft-mtp uses the model's native MTP head)
    "mtp_n1": {
        "LLAMA_SPEC_TYPE": "draft-mtp",
        "LLAMA_SPEC_DRAFT_N_MAX": "1",
    },
    "mtp_n2": {
        "LLAMA_SPEC_TYPE": "draft-mtp",
        "LLAMA_SPEC_DRAFT_N_MAX": "2",
    },
    "mtp_n3": {
        "LLAMA_SPEC_TYPE": "draft-mtp",
        "LLAMA_SPEC_DRAFT_N_MAX": "3",
    },

    # Ngram-mod only (prompt-lookup speculation, CPU-only)
    "ngram_only": {
        "LLAMA_SPEC_TYPE": "ngram-mod",
        "LLAMA_SPEC_DRAFT_N_MAX": "3",
    },

    # MTP + ngram-mod chained (MTP first, ngram as fallback)
    "mtp_ngram_n2": {
        "LLAMA_SPEC_TYPE": "ngram-mod,draft-mtp",
        "LLAMA_SPEC_DRAFT_N_MAX": "2",
    },
    "mtp_ngram_n3": {
        "LLAMA_SPEC_TYPE": "ngram-mod,draft-mtp",
        "LLAMA_SPEC_DRAFT_N_MAX": "3",
    },
}

# KV cache dtype configs for Phase 2
# llama.cpp uses separate cache-type-k and cache-type-v
KV_CONFIGS: dict[str, dict[str, str]] = {
    "turbo4_turbo2": {"LLAMA_KV_K": "turbo4", "LLAMA_KV_V": "turbo2"},
    "turbo3_turbo2": {"LLAMA_KV_K": "turbo3", "LLAMA_KV_V": "turbo2"},
    "turbo4_turbo1": {"LLAMA_KV_K": "turbo4", "LLAMA_KV_V": "turbo1"},
    "turbo3_turbo1": {"LLAMA_KV_K": "turbo3", "LLAMA_KV_V": "turbo1"},
    "q8_0_q8_0":    {"LLAMA_KV_K": "q8_0",   "LLAMA_KV_V": "q8_0"},
    "q4_0_q4_0":    {"LLAMA_KV_K": "q4_0",   "LLAMA_KV_V": "q4_0"},
    "f16_f16":       {"LLAMA_KV_K": "f16",    "LLAMA_KV_V": "f16"},
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
    kv_k: str = DEFAULT_KV_K,
    kv_v: str = DEFAULT_KV_V,
    spec_type: str = DEFAULT_SPEC,
    spec_n_max: str = "3",
    extra_env: dict[str, str] | None = None,
) -> None:
    """Force-recreate llama-server with env overrides via models.ini injection.

    llama.cpp reads config from models.ini, not env vars. For benchmarking we
    pass env vars that entrypoint.sh maps to CLI args. For configs that require
    models.ini changes, we sed-replace the relevant lines.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Pass overrides that entrypoint.sh or docker-compose can pick up
    env["LLAMA_SPEC_TYPE"] = spec_type
    env["LLAMA_SPEC_DRAFT_N_MAX"] = spec_n_max
    env["LLAMA_KV_K"] = kv_k
    env["LLAMA_KV_V"] = kv_v

    cmd = [
        "docker", "compose", "-f", COMPOSE_FILE,
        "up", "-d", "--force-recreate", "llama-server",
    ]
    _log(f"Restarting server: kv_k={kv_k} kv_v={kv_v} spec={spec_type} n_max={spec_n_max}")
    result = subprocess.run(
        cmd, env=env, cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
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
    """Stop llama-server (but not other services)."""
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "stop", "llama-server"],
        cwd=PROJECT_ROOT, capture_output=True,
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def read_api_key() -> str:
    """Read LLAMA_API_KEY from .env if it exists."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("LLAMA_API_KEY="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            return val
    return ""


def run_scenario(
    scenario: Scenario,
    dry: bool = False,
    api_key: str = "",
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """Execute one benchmark run via streaming chat completion. Returns metrics."""
    messages = [
        {"role": "system", "content": scenario.system},
        {"role": "user", "content": scenario.user},
    ]

    body: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "stream": True,
    }

    if dry:
        body.update(DRY_ON_PARAMS)

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

                # Usage chunk
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

    # llama.cpp reports timings in /health or in response headers;
    # for streaming we compute from wall-clock.
    return {
        "ttft_s": round(ttft, 3),
        "decode_tps": round(completion_tokens / decode_time, 2) if decode_time > 0 else 0,
        "overall_tps": round(completion_tokens / total_time, 2) if total_time > 0 else 0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": round(total_time, 3),
        "chunks": chunks_received,
    }


def _run_single_stream(
    messages: list[dict], max_tokens: int, dry: bool, api_key: str,
) -> dict[str, Any]:
    """Single streaming request with custom messages/max_tokens (for stress tests)."""
    body: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stream": True,
    }
    if dry:
        body.update(DRY_ON_PARAMS)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t_start = time.monotonic()
    t_first: float | None = None
    t_last = t_start
    completion_tokens = 0
    prompt_tokens = 0

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        with client.stream("POST", CHAT_URL, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"API {resp.status_code}")
            for raw_line in resp.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                choices = chunk.get("choices", [])
                if choices and (choices[0].get("delta", {}).get("content")):
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now

    total = t_last - t_start
    ttft = (t_first - t_start) if t_first else total
    decode = (t_last - t_first) if t_first else 0.001

    return {
        "ttft_s": round(ttft, 3),
        "decode_tps": round(completion_tokens / decode, 2) if decode > 0 else 0,
        "overall_tps": round(completion_tokens / total, 2) if total > 0 else 0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": round(total, 3),
    }


def _run_parallel_requests(
    prompts: list[dict], max_tokens: int, dry: bool, api_key: str,
) -> dict[str, Any]:
    """Run N requests concurrently, return aggregate metrics."""
    t_start = time.monotonic()

    def _one(p: dict) -> dict:
        msgs = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        return _run_single_stream(msgs, max_tokens, dry, api_key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(_one, p) for p in prompts]
        results = [f.result() for f in futures]

    wall = time.monotonic() - t_start
    agg_tps = sum(r["completion_tokens"] for r in results) / wall if wall > 0 else 0
    per_tps = [r["overall_tps"] for r in results]

    return {
        "wall_time_s": round(wall, 3),
        "aggregate_tps": round(agg_tps, 2),
        "per_request_tps_mean": round(statistics.mean(per_tps), 2) if per_tps else 0,
        "per_request_tps_min": round(min(per_tps), 2) if per_tps else 0,
        "per_request_results": results,
    }


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def flush_result(result: dict[str, Any], path: Path) -> None:
    """Append one result to JSONL immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load all results from JSONL. Lines starting with # are comments."""
    if not path.exists():
        return []
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            results.append(json.loads(line))
    return results


def result_key(r: dict) -> tuple:
    """Unique key for deduplication / resume."""
    sc = r.get("server_config", {})
    rc = r.get("request_config", {})
    return (
        r.get("phase"),
        sc.get("kv_k"),
        sc.get("kv_v"),
        sc.get("spec_type"),
        rc.get("dry"),
        r.get("scenario"),
        r.get("run"),
    )


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _pick_winner(results: list[dict], config_key: str) -> str:
    """Pick best config by mean overall_tps across all scenarios."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return ""
    groups: dict[str, list[float]] = {}
    for r in ok:
        label = r["server_config"].get(config_key, r["server_config"].get("config_label", ""))
        tps = r["metrics"].get("overall_tps", 0)
        groups.setdefault(label, []).append(tps)
    ranked = sorted(groups.items(), key=lambda x: statistics.mean(x[1]), reverse=True)
    winner = ranked[0][0]
    _log(f"Winner ({config_key}): {winner} ({statistics.mean(ranked[0][1]):.1f} tok/s)")
    return winner


def _flush_remaining_as_error(
    phase: int, config: dict, label: str, dry: bool,
    runs_per_scenario: int, completed: set[tuple],
    results_path: Path, all_results: list[dict[str, Any]],
    tracker: ProgressTracker, error_msg: str,
) -> None:
    """Record all un-completed runs for this config as errors."""
    for scenario in SCENARIOS:
        for run_num in range(1, runs_per_scenario + 1):
            key = (phase, config.get("kv_k", ""), config.get("kv_v", ""),
                   config.get("spec_type", ""), dry, scenario.name, run_num)
            if key in completed:
                continue
            completed.add(key)
            result = {
                "phase": phase,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_config": config,
                "request_config": {"dry": dry},
                "scenario": scenario.name,
                "run": run_num,
                "metrics": {},
                "status": "error",
                "error": error_msg,
            }
            flush_result(result, results_path)
            all_results.append(result)
            tracker.update(f"{label} x {scenario.name} run {run_num} (server crashed)")


def _run_config_sweep(
    *,
    phase: int,
    phase_name: str,
    configs: list[dict[str, Any]],
    dry: bool,
    runs_per_scenario: int,
    results_path: Path,
    completed: set[tuple],
    api_key: str,
    stop_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Generic sweep: restart per config, run all scenarios x runs."""
    total = len(configs) * len(SCENARIOS) * runs_per_scenario
    tracker = ProgressTracker(total, f"Phase {phase}: {phase_name}")
    all_results: list[dict[str, Any]] = []
    last_restart_key: tuple | None = None

    for cfg in configs:
        label = cfg["label"]
        kv_k = cfg.get("kv_k", DEFAULT_KV_K)
        kv_v = cfg.get("kv_v", DEFAULT_KV_V)
        spec_type = cfg.get("spec_type", DEFAULT_SPEC)
        spec_n_max = cfg.get("spec_n_max", "3")

        restart_key = (kv_k, kv_v, spec_type, spec_n_max)
        server_config = {
            "kv_k": kv_k, "kv_v": kv_v,
            "spec_type": spec_type, "spec_n_max": spec_n_max,
            "config_label": label,
        }

        # Skip entire config if all runs cached
        all_cached = all(
            (phase, kv_k, kv_v, spec_type, dry, sc.name, rn) in completed
            for sc in SCENARIOS for rn in range(1, runs_per_scenario + 1)
        )
        if all_cached:
            _log(f"Skipping {label} — all runs cached")
            for sc in SCENARIOS:
                for rn in range(1, runs_per_scenario + 1):
                    tracker.update(f"{label} x {sc.name} run {rn} (cached)")
            continue

        # Only restart if server config changed
        if restart_key != last_restart_key:
            try:
                restart_server(kv_k, kv_v, spec_type, spec_n_max)
                wait_for_health()
                last_restart_key = restart_key
            except (TimeoutError, RuntimeError) as e:
                _log(f"Server failed for {label}: {e}")
                last_restart_key = None
                if stop_on_error:
                    tracker.finish()
                    raise
                _flush_remaining_as_error(
                    phase, server_config, label, dry, runs_per_scenario,
                    completed, results_path, all_results, tracker,
                    f"server_startup_failed: {e}",
                )
                continue

            # Warmup (discard)
            try:
                run_scenario(SCENARIOS[0], dry=False, api_key=api_key)
            except Exception as e:
                _log(f"Warmup failed ({e}), continuing anyway")
        else:
            _log(f"Server already running with {label} — skipping restart")

        server_crashed = False
        for scenario in SCENARIOS:
            if server_crashed:
                break
            for run_num in range(1, runs_per_scenario + 1):
                key = (phase, kv_k, kv_v, spec_type, dry, scenario.name, run_num)
                if key in completed:
                    tracker.update(f"{label} x {scenario.name} run {run_num} (cached)")
                    continue

                run_start = time.monotonic()
                try:
                    metrics = run_scenario(scenario, dry=dry, api_key=api_key)
                    status = "ok"
                    error = None
                except (httpx.ConnectError, httpx.RemoteProtocolError,
                        ConnectionError) as e:
                    _log(f"\nSERVER CRASHED: {label} x {scenario.name} run {run_num}: {e}")
                    if stop_on_error:
                        tracker.finish()
                        raise RuntimeError(f"server crashed: {e}") from e
                    server_crashed = True
                    metrics = {}
                    status = "error"
                    error = f"server_crashed: {e}"
                except Exception as e:
                    _log(f"\nERROR: {label} x {scenario.name} run {run_num}: {e}")
                    if stop_on_error:
                        tracker.finish()
                        raise
                    metrics = {}
                    status = "error"
                    error = str(e)

                result = {
                    "phase": phase,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": server_config,
                    "request_config": {"dry": dry},
                    "scenario": scenario.name,
                    "run": run_num,
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                }
                flush_result(result, results_path)
                all_results.append(result)
                completed.add(key)

                run_duration = time.monotonic() - run_start
                tps = metrics.get("overall_tps") if metrics else None
                tracker.update(
                    f"{label} x {scenario.name} run {run_num}/{runs_per_scenario}",
                    last_tps=tps, duration=run_duration,
                )
                if server_crashed:
                    break

        if server_crashed:
            last_restart_key = None
            _flush_remaining_as_error(
                phase, server_config, label, dry, runs_per_scenario,
                completed, results_path, all_results, tracker,
                "server_crashed: container exited mid-batch",
            )

    tracker.finish()
    return all_results


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def run_phase_1(
    runs: int, results_path: Path, completed: set[tuple], api_key: str,
    *, stop_on_error: bool = False,
) -> dict[str, str]:
    """Phase 1: Speculative decoding sweep. Returns winner config."""
    _log("\n=== Phase 1: Speculative Decoding ===")
    configs = []
    for label, env_map in SPEC_CONFIGS.items():
        configs.append({
            "label": label,
            "spec_type": env_map.get("LLAMA_SPEC_TYPE", DEFAULT_SPEC),
            "spec_n_max": env_map.get("LLAMA_SPEC_DRAFT_N_MAX", "3"),
        })
    _run_config_sweep(
        phase=1, phase_name="Speculative", configs=configs,
        dry=False, runs_per_scenario=runs, results_path=results_path,
        completed=completed, api_key=api_key, stop_on_error=stop_on_error,
    )
    all_p1 = [r for r in load_results(results_path) if r.get("phase") == 1]
    winner_label = _pick_winner(all_p1, "config_label")
    # Return the full config dict for the winner
    return SPEC_CONFIGS.get(winner_label, {
        "LLAMA_SPEC_TYPE": DEFAULT_SPEC,
        "LLAMA_SPEC_DRAFT_N_MAX": "3",
    })


def run_phase_2(
    runs: int, results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: dict[str, str], *, stop_on_error: bool = False,
) -> dict[str, str]:
    """Phase 2: KV cache dtype sweep. Returns winner config."""
    _log("\n=== Phase 2: KV Cache Dtype ===")
    configs = []
    for label, kv_map in KV_CONFIGS.items():
        configs.append({
            "label": label,
            "kv_k": kv_map["LLAMA_KV_K"],
            "kv_v": kv_map["LLAMA_KV_V"],
            "spec_type": spec_winner.get("LLAMA_SPEC_TYPE", DEFAULT_SPEC),
            "spec_n_max": spec_winner.get("LLAMA_SPEC_DRAFT_N_MAX", "3"),
        })
    _run_config_sweep(
        phase=2, phase_name="KV Cache", configs=configs,
        dry=False, runs_per_scenario=runs, results_path=results_path,
        completed=completed, api_key=api_key, stop_on_error=stop_on_error,
    )
    all_p2 = [r for r in load_results(results_path) if r.get("phase") == 2]
    winner_label = _pick_winner(all_p2, "config_label")
    return KV_CONFIGS.get(winner_label, {
        "LLAMA_KV_K": DEFAULT_KV_K,
        "LLAMA_KV_V": DEFAULT_KV_V,
    })


def run_phase_3(
    runs: int, results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: dict, kv_winner: dict, *, stop_on_error: bool = False,
) -> bool:
    """Phase 3: DRY on vs off. Returns True if DRY is recommended."""
    _log("\n=== Phase 3: DRY Sampling ===")

    kv_k = kv_winner.get("LLAMA_KV_K", DEFAULT_KV_K)
    kv_v = kv_winner.get("LLAMA_KV_V", DEFAULT_KV_V)
    spec = spec_winner.get("LLAMA_SPEC_TYPE", DEFAULT_SPEC)
    spec_n = spec_winner.get("LLAMA_SPEC_DRAFT_N_MAX", "3")

    # Restart once with winning config
    restart_server(kv_k, kv_v, spec, spec_n)
    wait_for_health()
    run_scenario(SCENARIOS[0], dry=False, api_key=api_key)  # warmup

    total = 2 * len(SCENARIOS) * runs
    tracker = ProgressTracker(total, "Phase 3: DRY")

    for dry in [False, True]:
        dry_label = "dry_on" if dry else "dry_off"
        for scenario in SCENARIOS:
            for run_num in range(1, runs + 1):
                key = (3, kv_k, kv_v, spec, dry, scenario.name, run_num)
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
                    if stop_on_error:
                        tracker.finish()
                        raise
                    metrics = {}
                    status = "error"
                    error = str(e)

                result = {
                    "phase": 3,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_k": kv_k, "kv_v": kv_v,
                        "spec_type": spec, "spec_n_max": spec_n,
                        "config_label": dry_label,
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
                    last_tps=tps, duration=run_duration,
                )

    tracker.finish()

    # Determine winner
    p3 = [r for r in load_results(results_path) if r.get("phase") == 3 and r.get("status") == "ok"]
    dry_off = [r["metrics"]["overall_tps"] for r in p3 if not r["request_config"]["dry"]]
    dry_on = [r["metrics"]["overall_tps"] for r in p3 if r["request_config"]["dry"]]
    if dry_off and dry_on:
        mean_off, mean_on = statistics.mean(dry_off), statistics.mean(dry_on)
        # Prefer DRY on when throughput cost <= 3% (quality benefit)
        dry_winner = mean_on >= mean_off * 0.97
        _log(f"DRY winner: {'on' if dry_winner else 'off'} "
             f"(off={mean_off:.1f}, on={mean_on:.1f} tok/s)")
        return dry_winner
    return False


def run_phase_4(
    results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: dict, kv_winner: dict, *, stop_on_error: bool = False,
) -> None:
    """Phase 4: Cross-validate top 3 full configs with 5 runs each."""
    _log("\n=== Phase 4: Cross-validation ===")

    all_results = load_results(results_path)
    if not all_results:
        _log("No results to cross-validate.")
        return

    combo_tps: dict[tuple, list[float]] = {}
    for r in all_results:
        if r.get("status") != "ok":
            continue
        sc = r["server_config"]
        rc = r["request_config"]
        combo = (sc.get("kv_k", ""), sc.get("kv_v", ""), sc.get("spec_type", ""), rc["dry"])
        tps = r["metrics"].get("overall_tps", 0)
        combo_tps.setdefault(combo, []).append(tps)

    ranked = sorted(combo_tps.items(), key=lambda x: statistics.mean(x[1]), reverse=True)
    top3 = ranked[:3]
    _log("Top 3 configs for cross-validation:")
    for (kk, kv, sp, dry), tps_list in top3:
        _log(f"  kv={kk}/{kv} spec={sp or '(none)'} dry={dry} -> {statistics.mean(tps_list):.1f} tok/s")

    crossval_runs = 5
    total = len(top3) * len(SCENARIOS) * crossval_runs
    tracker = ProgressTracker(total, "Phase 4: Cross-val")

    last_key: tuple | None = None
    for (kk, kv, sp, dry), _ in top3:
        restart_key = (kk, kv, sp)
        if restart_key != last_key:
            restart_server(kk, kv, sp, spec_winner.get("LLAMA_SPEC_DRAFT_N_MAX", "3"))
            wait_for_health()
            run_scenario(SCENARIOS[0], dry=False, api_key=api_key)
            last_key = restart_key

        label = f"kv={kk}/{kv} spec={sp[:10]} dry={dry}"
        for scenario in SCENARIOS:
            for run_num in range(1, crossval_runs + 1):
                key = (4, kk, kv, sp, dry, scenario.name, run_num)
                if key in completed:
                    tracker.update(f"{label} x {scenario.name} run {run_num} (cached)")
                    continue

                run_start = time.monotonic()
                try:
                    metrics = run_scenario(scenario, dry=dry, api_key=api_key)
                    status = "ok"
                    error = None
                except Exception as e:
                    if stop_on_error:
                        tracker.finish()
                        raise
                    metrics = {}
                    status = "error"
                    error = str(e)

                result = {
                    "phase": 4,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_k": kk, "kv_v": kv,
                        "spec_type": sp, "config_label": label,
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
                    last_tps=tps, duration=run_duration,
                )

    tracker.finish()


# ---------------------------------------------------------------------------
# Phase 5: Stress tests
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "sustained_2k",
        "description": "Sustained 2048-token generation",
        "max_tokens": 2048,
        "system": "You are a detailed technical writer.",
        "user": (
            "Write a comprehensive tutorial on building a distributed task queue "
            "system from scratch in Python. Cover architecture decisions, message "
            "serialization, worker lifecycle, failure recovery, exactly-once delivery, "
            "monitoring, and deployment. Include full code examples for each component."
        ),
    },
    {
        "name": "sustained_4k",
        "description": "Sustained 4096-token generation",
        "max_tokens": 4096,
        "system": "You are an expert software architect.",
        "user": (
            "Design and fully implement a real-time collaborative text editor backend "
            "in Python using operational transforms. Include: the OT algorithm with "
            "transform functions for insert/delete, a server that manages document state "
            "and client connections via WebSockets, conflict resolution logic, undo/redo "
            "support, persistence layer, and comprehensive test suite. Write all the code."
        ),
    },
    {
        "name": "rapid_short_burst",
        "description": "10 rapid short requests back-to-back",
        "max_tokens": 128,
        "system": "Answer in one sentence.",
        "user": "What is {topic}?",
        "burst": 10,
        "burst_topics": [
            "quantum computing", "photosynthesis", "the Turing test",
            "CRISPR gene editing", "blockchain consensus", "neural plasticity",
            "dark matter", "the halting problem", "RISC-V architecture", "mRNA vaccines",
        ],
    },
    {
        "name": "parallel_2_slots",
        "description": "2 concurrent requests (parallel=2 in models.ini)",
        "max_tokens": 1024,
        "parallel": 2,
        "prompts": [
            {
                "system": "You are an expert Python developer.",
                "user": "Write a complete async HTTP client library with connection pooling, retry logic, and rate limiting.",
            },
            {
                "system": "You are a database expert.",
                "user": "Write a complete connection pool implementation for PostgreSQL in Python with health checks.",
            },
        ],
    },
    {
        "name": "parallel_3_slots",
        "description": "3 concurrent requests (parallel=3 in models.ini)",
        "max_tokens": 512,
        "parallel": 3,
        "prompts": [
            {"system": "You are a Python developer.", "user": "Write a binary search tree with insert/delete/search."},
            {"system": "You are a Go developer.", "user": "Write a concurrent-safe LRU cache in Go."},
            {"system": "You are a Rust developer.", "user": "Write a thread-safe ring buffer in Rust."},
        ],
    },
]


def run_phase_5(
    results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: dict, kv_winner: dict, dry_winner: bool,
    *, stop_on_error: bool = False,
) -> None:
    """Phase 5: Stress tests — sustained generation, bursts, parallel slots."""
    _log("\n=== Phase 5: Stress Tests ===")

    kv_k = kv_winner.get("LLAMA_KV_K", DEFAULT_KV_K)
    kv_v = kv_winner.get("LLAMA_KV_V", DEFAULT_KV_V)
    spec = spec_winner.get("LLAMA_SPEC_TYPE", DEFAULT_SPEC)
    spec_n = spec_winner.get("LLAMA_SPEC_DRAFT_N_MAX", "3")

    restart_server(kv_k, kv_v, spec, spec_n)
    wait_for_health()

    stress_runs = 3
    total = sum(
        (s.get("burst", 1) if s.get("burst") else 1) * stress_runs
        if not s.get("parallel") else stress_runs
        for s in STRESS_SCENARIOS
    )
    tracker = ProgressTracker(total, "Phase 5: Stress")

    for stress in STRESS_SCENARIOS:
        is_parallel = stress.get("parallel", 0) > 0
        is_burst = stress.get("burst", 0) > 0

        for run_num in range(1, stress_runs + 1):
            if is_parallel:
                try:
                    metrics = _run_parallel_requests(
                        stress["prompts"], stress["max_tokens"], dry_winner, api_key,
                    )
                    status = "ok"
                    error = None
                except Exception as e:
                    metrics = {}
                    status = "error"
                    error = str(e)
                result = {
                    "phase": 5,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_k": kv_k, "kv_v": kv_v,
                        "spec_type": spec, "config_label": stress["name"],
                    },
                    "request_config": {"dry": dry_winner},
                    "scenario": stress["name"],
                    "run": run_num,
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                    "parallel_n": stress["parallel"],
                }
                flush_result(result, results_path)
                tracker.update(f"{stress['name']} run {run_num}",
                               last_tps=metrics.get("aggregate_tps"))
                continue

            iterations = stress.get("burst", 1) if is_burst else 1
            for burst_i in range(iterations):
                user_msg = stress["user"]
                if is_burst and stress.get("burst_topics"):
                    user_msg = user_msg.format(topic=stress["burst_topics"][burst_i])
                messages = [
                    {"role": "system", "content": stress["system"]},
                    {"role": "user", "content": user_msg},
                ]
                try:
                    metrics = _run_single_stream(
                        messages, stress["max_tokens"], dry_winner, api_key,
                    )
                    status = "ok"
                    error = None
                except Exception as e:
                    metrics = {}
                    status = "error"
                    error = str(e)
                result = {
                    "phase": 5,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_k": kv_k, "kv_v": kv_v,
                        "spec_type": spec, "config_label": stress["name"],
                    },
                    "request_config": {"dry": dry_winner},
                    "scenario": stress["name"],
                    "run": run_num,
                    "burst_index": burst_i if is_burst else None,
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                }
                flush_result(result, results_path)
                tracker.update(
                    f"{stress['name']} run {run_num}" +
                    (f" burst {burst_i+1}/{iterations}" if is_burst else ""),
                    last_tps=metrics.get("overall_tps"),
                )

    tracker.finish()


# ---------------------------------------------------------------------------
# Live mode (no restarts — benchmark against running server)
# ---------------------------------------------------------------------------

def run_live(
    runs: int, api_key: str, *, dry: bool = False, include_stress: bool = False,
) -> None:
    """Benchmark against a running server. No docker restarts, terminal output only."""
    _log("=== Live Benchmark ===")
    _log(f"Runs per scenario: {runs}")
    _log(f"DRY: {'on' if dry else 'off'}")

    # Warmup (discard)
    _log("Warmup run...")
    try:
        run_scenario(SCENARIOS[0], dry=False, api_key=api_key)
    except Exception as e:
        _log(f"Warmup failed: {e}")

    results: list[dict[str, Any]] = []
    total = len(SCENARIOS) * runs
    tracker = ProgressTracker(total, "Live")

    for scenario in SCENARIOS:
        for run_num in range(1, runs + 1):
            run_start = time.monotonic()
            try:
                metrics = run_scenario(scenario, dry=dry, api_key=api_key)
                status = "ok"
                error = None
            except Exception as e:
                _log(f"\nERROR: {scenario.name} run {run_num}: {e}")
                metrics = {}
                status = "error"
                error = str(e)

            results.append({
                "scenario": scenario.name,
                "run": run_num,
                "metrics": metrics,
                "status": status,
                "error": error,
            })

            run_duration = time.monotonic() - run_start
            tps = metrics.get("overall_tps") if metrics else None
            tracker.update(
                f"{scenario.name} run {run_num}/{runs}",
                last_tps=tps, duration=run_duration,
            )

    tracker.finish()

    # Optional stress
    stress_results: list[dict[str, Any]] = []
    if include_stress:
        stress_total = sum(
            (s.get("burst", 1) if s.get("burst") else 1) * 3
            if not s.get("parallel") else 3
            for s in STRESS_SCENARIOS
        )
        stress_tracker = ProgressTracker(stress_total, "Live-Stress")

        for stress in STRESS_SCENARIOS:
            is_parallel = stress.get("parallel", 0) > 0
            is_burst = stress.get("burst", 0) > 0

            for run_num in range(1, 4):
                if is_parallel:
                    try:
                        metrics = _run_parallel_requests(
                            stress["prompts"], stress["max_tokens"], dry, api_key,
                        )
                        status = "ok"
                        error = None
                    except Exception as e:
                        metrics = {}
                        status = "error"
                        error = str(e)
                    stress_results.append({
                        "scenario": stress["name"], "run": run_num,
                        "metrics": metrics, "status": status, "error": error,
                        "parallel_n": stress["parallel"],
                    })
                    stress_tracker.update(f"{stress['name']} run {run_num}",
                                          last_tps=metrics.get("aggregate_tps"))
                    continue

                iterations = stress.get("burst", 1) if is_burst else 1
                for burst_i in range(iterations):
                    user_msg = stress["user"]
                    if is_burst and stress.get("burst_topics"):
                        user_msg = user_msg.format(topic=stress["burst_topics"][burst_i])
                    messages = [
                        {"role": "system", "content": stress["system"]},
                        {"role": "user", "content": user_msg},
                    ]
                    try:
                        metrics = _run_single_stream(messages, stress["max_tokens"], dry, api_key)
                        status = "ok"
                        error = None
                    except Exception as e:
                        metrics = {}
                        status = "error"
                        error = str(e)
                    stress_results.append({
                        "scenario": stress["name"], "run": run_num,
                        "burst_index": burst_i if is_burst else None,
                        "metrics": metrics, "status": status, "error": error,
                    })
                    stress_tracker.update(
                        f"{stress['name']} run {run_num}" +
                        (f" burst {burst_i+1}/{iterations}" if is_burst else ""),
                        last_tps=metrics.get("overall_tps"),
                    )

        stress_tracker.finish()

    _print_terminal_report(results, stress_results)


def _print_terminal_report(
    results: list[dict[str, Any]],
    stress_results: list[dict[str, Any]] | None = None,
) -> None:
    """Print results table to stdout."""
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]

    if not ok:
        print("\nNo successful runs.")
        return

    scenarios = sorted(set(r["scenario"] for r in ok))

    rows: list[tuple] = []
    for sc in scenarios:
        sc_ok = [r for r in ok if r["scenario"] == sc]
        decode_vals = [r["metrics"]["decode_tps"] for r in sc_ok]
        overall_vals = [r["metrics"]["overall_tps"] for r in sc_ok]
        ttft_vals = [r["metrics"]["ttft_s"] for r in sc_ok]
        tok_vals = [r["metrics"]["completion_tokens"] for r in sc_ok]

        d_mean = statistics.mean(decode_vals)
        d_std = statistics.stdev(decode_vals) if len(decode_vals) > 1 else 0.0
        o_mean = statistics.mean(overall_vals)
        o_std = statistics.stdev(overall_vals) if len(overall_vals) > 1 else 0.0
        t_mean = statistics.mean(ttft_vals)
        avg_tok = round(statistics.mean(tok_vals))

        rows.append((sc, d_mean, d_std, o_mean, o_std, t_mean, avg_tok, len(sc_ok)))

    print()
    print("=" * 96)
    print("  LIVE BENCHMARK RESULTS")
    print("=" * 96)
    print(f"  {'Scenario':<24} {'Decode tok/s':>14} {'Overall tok/s':>15} {'TTFT(s)':>9} {'Tokens':>7} {'Runs':>5}")
    print("-" * 96)

    all_decode, all_overall, all_ttft = [], [], []
    for (sc, d_m, d_s, o_m, o_s, t_m, tok, n) in rows:
        print(f"  {sc:<24} {d_m:>7.1f}\u00b1{d_s:<5.1f} {o_m:>8.1f}\u00b1{o_s:<5.1f} {t_m:>8.3f} {tok:>7} {n:>5}")
        all_decode.append(d_m)
        all_overall.append(o_m)
        all_ttft.append(t_m)

    print("-" * 96)
    d_total = statistics.mean(all_decode)
    o_total = statistics.mean(all_overall)
    t_total = statistics.mean(all_ttft)
    total_runs = sum(r[7] for r in rows)
    print(f"  {'MEAN':<24} {d_total:>7.1f}       {o_total:>8.1f}       {t_total:>8.3f}         {total_runs:>5}")
    print("=" * 96)

    if errors:
        print(f"\n  ERRORS: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e['scenario']} run {e['run']}: {e.get('error', '?')[:80]}")

    # Stress results
    if stress_results:
        s_ok = [r for r in stress_results if r["status"] == "ok"]
        if s_ok:
            seq = [r for r in s_ok if not r.get("parallel_n")]
            par = [r for r in s_ok if r.get("parallel_n")]

            if seq:
                seq_scenarios = sorted(set(r["scenario"] for r in seq))
                print()
                print("  STRESS \u2014 SEQUENTIAL")
                print("-" * 96)
                print(f"  {'Test':<28} {'tok/s (mean)':>13} {'min':>8} {'max':>8} {'TTFT(s)':>9} {'Runs':>5}")
                print("-" * 96)
                for sc in seq_scenarios:
                    sc_runs = [r for r in seq if r["scenario"] == sc]
                    tps = [r["metrics"]["overall_tps"] for r in sc_runs if r["metrics"].get("overall_tps")]
                    ttft = [r["metrics"]["ttft_s"] for r in sc_runs if r["metrics"].get("ttft_s")]
                    if tps:
                        print(f"  {sc:<28} {statistics.mean(tps):>10.1f}   {min(tps):>8.1f} {max(tps):>8.1f} {statistics.mean(ttft):>8.3f} {len(sc_runs):>5}")

            if par:
                par_scenarios = sorted(set(r["scenario"] for r in par))
                print()
                print("  STRESS \u2014 PARALLEL")
                print("-" * 96)
                print(f"  {'Test':<28} {'Slots':>5} {'Agg tok/s':>10} {'Per-req':>9} {'Wall(s)':>9} {'Runs':>5}")
                print("-" * 96)
                for sc in par_scenarios:
                    sc_runs = [r for r in par if r["scenario"] == sc]
                    n = sc_runs[0].get("parallel_n", "?")
                    agg = [r["metrics"]["aggregate_tps"] for r in sc_runs if r["metrics"].get("aggregate_tps")]
                    per = [r["metrics"]["per_request_tps_mean"] for r in sc_runs if r["metrics"].get("per_request_tps_mean")]
                    wall = [r["metrics"]["wall_time_s"] for r in sc_runs if r["metrics"].get("wall_time_s")]
                    if agg:
                        print(f"  {sc:<28} {n:>5} {statistics.mean(agg):>10.1f} {statistics.mean(per):>9.1f} {statistics.mean(wall):>9.1f} {len(sc_runs):>5}")
                print("=" * 96)

    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    sys.stderr.write(f"\n[{ts}] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="llama.cpp benchmark orchestrator")
    parser.add_argument("--runs", type=int, default=3, help="Runs per scenario (default: 3)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Run only this phase")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report from JSONL")
    parser.add_argument("--results-file", type=str, help="Path to existing JSONL (for resume/report)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort immediately on any error")
    parser.add_argument("--live", action="store_true",
                        help="Benchmark against running server (no restarts)")
    parser.add_argument("--dry", action="store_true",
                        help="Send DRY sampling params (use with --live)")
    parser.add_argument("--stress", action="store_true",
                        help="Include stress scenarios (use with --live)")
    args = parser.parse_args()

    # Live mode
    if args.live:
        api_key = read_api_key()
        run_live(args.runs, api_key, dry=args.dry, include_stress=args.stress)
        return

    # Determine results file
    if args.results_file:
        results_path = Path(args.results_file)
    elif args.resume or args.report_only:
        existing = sorted(RESULTS_DIR.glob("*_runs.jsonl"))
        if not existing:
            _log("ERROR: no existing results found")
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
        _log(f"Report: {report_path}")
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

    spec_winner = {"LLAMA_SPEC_TYPE": DEFAULT_SPEC, "LLAMA_SPEC_DRAFT_N_MAX": "3"}
    kv_winner = {"LLAMA_KV_K": DEFAULT_KV_K, "LLAMA_KV_V": DEFAULT_KV_V}
    dry_winner = False

    if args.phase is None or args.phase == 1:
        spec_winner = run_phase_1(args.runs, results_path, completed, api_key,
                                   stop_on_error=args.stop_on_error)

    if args.phase is None or args.phase == 2:
        kv_winner = run_phase_2(args.runs, results_path, completed, api_key, spec_winner,
                                stop_on_error=args.stop_on_error)

    if args.phase is None or args.phase == 3:
        dry_winner = run_phase_3(args.runs, results_path, completed, api_key,
                                 spec_winner, kv_winner, stop_on_error=args.stop_on_error)

    if args.phase is None or args.phase == 4:
        run_phase_4(results_path, completed, api_key, spec_winner, kv_winner,
                    stop_on_error=args.stop_on_error)

    if args.phase is None or args.phase == 5:
        run_phase_5(results_path, completed, api_key, spec_winner, kv_winner, dry_winner,
                    stop_on_error=args.stop_on_error)

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


if __name__ == "__main__":
    main()
