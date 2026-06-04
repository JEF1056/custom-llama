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
    "mtp_n4": json.dumps({"method": "mtp", "num_speculative_tokens": 4}),

    # Ngram (GPU) — uses ngram_gpu for async scheduling + FULL cudagraph support
    # prompt_lookup_min >= 8 required for Qwen3 to avoid tool-call corruption
    # (see https://github.com/vllm-project/vllm/issues/40875)
    # NOTE: method="ngram" (CPU) disables async scheduling and forces PIECEWISE
    # cudagraphs, causing ~25-30% regression vs baseline. Use ngram_gpu instead.
    "ngram_tight": json.dumps({
        "method": "ngram_gpu", "num_speculative_tokens": 3,
        "prompt_lookup_max": 10, "prompt_lookup_min": 8,
    }),
    "ngram_default": json.dumps({
        "method": "ngram_gpu", "num_speculative_tokens": 5,
        "prompt_lookup_max": 15, "prompt_lookup_min": 8,
    }),
    "ngram_wide": json.dumps({
        "method": "ngram_gpu", "num_speculative_tokens": 7,
        "prompt_lookup_max": 20, "prompt_lookup_min": 8,
    }),

    # Ngram-first (GPU) + MTP configs are generated dynamically in Phase 1b
    # from the top 2 performing base configs (mtp + ngram).
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
    _log(f"Restarting server: kv={kv_cache_dtype} spec={speculative_config or '(none)'} dry={dry_config or '(none)'}")
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


def _scrape_raw_spec_counters() -> dict[str, float]:
    """Scrape raw cumulative counters from vLLM's Prometheus /metrics endpoint."""
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(METRICS_URL)
            if r.status_code != 200:
                return {}
    except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout, ConnectionError):
        return {}

    counters: dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        for name, key in [
            ("vllm:spec_decode_num_accepted_tokens_total", "accepted"),
            ("vllm:spec_decode_num_draft_tokens_total", "drafted"),
            ("vllm:spec_decode_num_emitted_tokens_total", "emitted"),
            # Underscore variants (metric naming varies by vLLM version)
            ("vllm_spec_decode_num_accepted_tokens_total", "accepted"),
            ("vllm_spec_decode_num_draft_tokens_total", "drafted"),
            ("vllm_spec_decode_num_emitted_tokens_total", "emitted"),
        ]:
            if line.startswith(name):
                try:
                    val = float(line.split()[-1])
                    if key not in counters:
                        counters[key] = val
                except (ValueError, IndexError):
                    pass
    return counters


def compute_spec_metrics(
    before: dict[str, float], after: dict[str, float],
) -> dict[str, float]:
    """Compute per-run speculative decoding metrics from before/after counter snapshots.

    Returns dict with acceptance_pct (0-100), accepted, drafted, emitted deltas.
    Returns empty dict if counters unavailable or spec decoding is off.
    """
    if not before or not after:
        return {}

    drafted_delta = after.get("drafted", 0) - before.get("drafted", 0)
    accepted_delta = after.get("accepted", 0) - before.get("accepted", 0)
    emitted_delta = after.get("emitted", 0) - before.get("emitted", 0)

    if drafted_delta <= 0:
        return {}  # no speculative tokens this run

    return {
        "acceptance_pct": round((accepted_delta / drafted_delta) * 100, 1),
        "accepted_tokens": int(accepted_delta),
        "drafted_tokens": int(drafted_delta),
        "emitted_tokens": int(emitted_delta),
    }


def run_scenario(
    scenario: Scenario,
    dry: bool = False,
    api_key: str = "",
) -> dict[str, Any]:
    """Execute one benchmark run via streaming chat completion. Returns metrics."""
    spec_before = _scrape_raw_spec_counters()

    messages = [
        {"role": "system", "content": scenario.system},
        {"role": "user", "content": scenario.user},
    ]

    body: dict[str, Any] = {
        "model": "qwen3.6-27b",
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
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

    # Compute per-run speculative decoding acceptance from counter deltas.
    # vLLM flushes Prometheus counters asynchronously — brief delay ensures
    # the /metrics endpoint reflects tokens from this request.
    time.sleep(1.0)
    spec_after = _scrape_raw_spec_counters()
    spec_metrics = compute_spec_metrics(spec_before, spec_after)
    if spec_metrics:
        result["spec_metrics"] = spec_metrics
    else:
        # Log raw values for debugging when no delta is detected
        if spec_before or spec_after:
            print(f"  [spec debug] before={spec_before} after={spec_after}")

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
        sc.get("kv_cache_dtype"),
        sc.get("speculative_config"),
        rc.get("dry"),
        r.get("scenario"),
        r.get("run"),
    )


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _flush_remaining_as_error(
    phase: int, kv: str, spec: str, label: str, dry: bool,
    runs_per_scenario: int, completed: set[tuple],
    results_path: Path, all_results: list[dict[str, Any]],
    tracker: ProgressTracker, error_msg: str,
) -> None:
    """Record all un-completed runs for this config as errors."""
    for scenario in SCENARIOS:
        for run_num in range(1, runs_per_scenario + 1):
            key = (phase, kv, spec, dry, scenario.name, run_num)
            if key in completed:
                continue
            # Check if already flushed this run (e.g. the failing run itself)
            completed.add(key)
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
                "metrics": {},
                "status": "error",
                "error": error_msg,
                "restart_duration_s": 0,
            }
            flush_result(result, results_path)
            all_results.append(result)
            tracker.update(f"{label} x {scenario.name} run {run_num} (server crashed)")


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
    last_kv, last_spec = None, None

    for cfg in configs:
        kv = cfg.get("kv_cache_dtype", fixed_kv)
        spec = cfg.get("speculative_config", fixed_spec)
        label = cfg[config_label_key]

        # Skip entire config if all its runs are already completed
        all_cached = all(
            (phase, kv, spec, dry, scenario.name, run_num) in completed
            for scenario in SCENARIOS
            for run_num in range(1, runs_per_scenario + 1)
        )
        if all_cached:
            _log(f"Skipping {label} — all runs cached")
            for scenario in SCENARIOS:
                for run_num in range(1, runs_per_scenario + 1):
                    tracker.update(f"{label} x {scenario.name} run {run_num} (cached)")
            continue

        # Only restart if server config changed
        if kv != last_kv or spec != last_spec:
            try:
                restart_server(kv, spec)
                restart_time_start = time.monotonic()
                wait_for_health()
                restart_duration = time.monotonic() - restart_time_start
                last_kv, last_spec = kv, spec
            except (TimeoutError, RuntimeError) as e:
                _log(f"Server failed for {label}: {e} — skipping config")
                last_kv, last_spec = None, None
                _flush_remaining_as_error(
                    phase, kv, spec, label, dry, runs_per_scenario,
                    completed, results_path, all_results, tracker,
                    f"server_startup_failed: {e}",
                )
                continue

            # Warmup — discard
            try:
                run_scenario(SCENARIOS[0], dry=False, api_key=api_key)
            except Exception as e:
                _log(f"Warmup failed ({e}), continuing anyway")
        else:
            restart_duration = 0.0
            _log(f"Server already running with kv={kv} spec={spec or '(none)'} — skipping restart")

        server_crashed = False
        for scenario in SCENARIOS:
            if server_crashed:
                break
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
                except (httpx.ConnectError, httpx.RemoteProtocolError,
                        ConnectionError) as e:
                    _log(f"\nSERVER CRASHED: {label} x {scenario.name} run {run_num}: {e}")
                    server_crashed = True
                    metrics = {}
                    status = "error"
                    error = f"server_crashed: {e}"
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
                completed.add(key)

                run_duration = time.monotonic() - run_start
                tps = metrics.get("overall_tps") if metrics else None
                tracker.update(
                    f"{label} x {scenario.name} run {run_num}/{runs_per_scenario}",
                    last_tps=tps,
                    duration=run_duration,
                )

                if server_crashed:
                    break

        # Server crashed mid-batch — flush remaining runs as errors, move on
        if server_crashed:
            last_kv, last_spec = None, None
            _flush_remaining_as_error(
                phase, kv, spec, label, dry, runs_per_scenario,
                completed, results_path, all_results, tracker,
                "server_crashed: container exited mid-batch",
            )

    tracker.finish()
    return all_results


def _build_ngram_first_configs(
    results: list[dict[str, Any]], top_n: int = 2,
) -> dict[str, str]:
    """Build ngram_first configs from the top-N base spec configs by decode_tps.

    Takes the best performing MTP and ngram configs, then creates ngram_first
    variants that combine GPU ngram pre-check with the winning MTP token counts.
    Only MTP-based configs can be used as the LLM proposer for ngram_first.
    """
    # Rank all successful base configs by mean decode_tps
    by_label: dict[str, list[float]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        label = r["server_config"].get("config_label", "")
        tps = r["metrics"].get("decode_tps", r["metrics"].get("overall_tps", 0))
        by_label.setdefault(label, []).append(tps)

    if not by_label:
        _log("WARNING: no results to build ngram_first configs from")
        return {}

    ranked = sorted(by_label.items(), key=lambda x: -statistics.mean(x[1]))
    _log("Base config ranking (decode_tps):")
    for label, tps_list in ranked:
        _log(f"  {label:20s}  {statistics.mean(tps_list):.1f} tok/s")

    # Pick top N configs, extract their spec JSON, build ngram_first variants
    configs: dict[str, str] = {}
    for label, _ in ranked:
        if len(configs) >= top_n:
            break
        # Find the spec config string for this label
        spec_str = SPEC_CONFIGS.get(label, "")
        if not spec_str:
            continue  # skip "none" baseline
        try:
            spec = json.loads(spec_str)
        except json.JSONDecodeError:
            continue

        # Only MTP method can be used as the LLM proposer for ngram_first
        method = spec.get("method", "")
        if method == "mtp":
            n = spec["num_speculative_tokens"]
            nf_label = f"ngram_mtp_n{n}"
            nf_config = {
                "method": "mtp",
                "num_speculative_tokens": n,
                "ngram_first": True,
                "ngram_first_gpu": True,
                "prompt_lookup_max": 10,
                "prompt_lookup_min": 8,
            }
            configs[nf_label] = json.dumps(nf_config)
            _log(f"  → ngram_first config: {nf_label}")
        elif method == "ngram_gpu":
            # Ngram-only can't be used as LLM proposer, but its performance
            # informs which MTP token count to pair with ngram_first.
            # Use its num_speculative_tokens as the MTP token count.
            n = spec["num_speculative_tokens"]
            nf_label = f"ngram_mtp_from_ngram_n{n}"
            nf_config = {
                "method": "mtp",
                "num_speculative_tokens": n,
                "ngram_first": True,
                "ngram_first_gpu": True,
                "prompt_lookup_max": spec.get("prompt_lookup_max", 10),
                "prompt_lookup_min": spec.get("prompt_lookup_min", 8),
            }
            configs[nf_label] = json.dumps(nf_config)
            _log(f"  → ngram_first config: {nf_label} (from ngram_gpu n={n})")

    return configs


def run_phase_1(
    runs: int, results_path: Path, completed: set[tuple], api_key: str,
) -> str:
    """Phase 1: Speculative decoding sweep. Returns winner spec config.

    Sub-phase 1a: Sweep base configs (MTP, ngram_gpu, baseline).
    Sub-phase 1b: Dynamically build ngram_first configs from top 2 performers,
                  then sweep those.
    """
    _log("\n=== Phase 1a: Base Speculative Decoding ===")
    base_configs = [
        {"speculative_config": v, "config_label": k}
        for k, v in SPEC_CONFIGS.items()
    ]
    base_results = _run_config_sweep(
        phase=1,
        phase_name="Speculative (base)",
        configs=base_configs,
        config_label_key="config_label",
        fixed_kv=DEFAULT_KV,
        fixed_spec="",
        dry=False,
        runs_per_scenario=runs,
        results_path=results_path,
        completed=completed,
        api_key=api_key,
    )

    # Include any cached Phase 1 results for ranking
    all_p1 = [r for r in load_results(results_path) if r.get("phase") == 1]

    # Phase 1b: Build ngram_first configs from top 2 base performers
    _log("\n=== Phase 1b: Ngram-First (dynamic from top 2) ===")
    ngram_first_configs = _build_ngram_first_configs(all_p1, top_n=2)

    nf_results: list[dict[str, Any]] = []
    if ngram_first_configs:
        nf_sweep = [
            {"speculative_config": v, "config_label": k}
            for k, v in ngram_first_configs.items()
        ]
        nf_results = _run_config_sweep(
            phase=1,
            phase_name="Speculative (ngram_first)",
            configs=nf_sweep,
            config_label_key="config_label",
            fixed_kv=DEFAULT_KV,
            fixed_spec="",
            dry=False,
            runs_per_scenario=runs,
            results_path=results_path,
            completed=completed,
            api_key=api_key,
        )
    else:
        _log("No ngram_first configs to test (no base results available)")

    # Pick overall winner from all Phase 1 results
    all_results = base_results + nf_results
    return _pick_winner(all_results, "speculative_config")


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
# Phase 5: Stress test
# ---------------------------------------------------------------------------

# Stress test scenarios — longer outputs and sustained generation
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
        "name": "long_context_response",
        "description": "Long input context + generation",
        "max_tokens": 1024,
        "system": "You are a thorough code reviewer.",
        "user": (
            "Review the following code and provide detailed feedback on every function, "
            "including bugs, performance issues, security vulnerabilities, and style "
            "improvements. Be extremely thorough.\n\n"
            + "```python\n"
            + "\n".join(
                f"def process_item_{i}(data: dict, config: dict) -> dict:\n"
                f"    result = {{}}\n"
                f"    for key in data:\n"
                f"        if config.get('transform_{i}'):\n"
                f"            result[key] = data[key] * {i + 1}\n"
                f"        else:\n"
                f"            result[key] = data[key]\n"
                f"    return result\n"
                for i in range(50)
            )
            + "```\n"
        ),
    },
    {
        "name": "rapid_short_burst",
        "description": "10 rapid short requests back-to-back",
        "max_tokens": 128,
        "system": "Answer in one sentence.",
        "user": "What is {topic}?",
        "burst": 10,  # special: run N times in quick succession
        "burst_topics": [
            "quantum computing", "photosynthesis", "the Turing test",
            "CRISPR gene editing", "blockchain consensus", "neural plasticity",
            "dark matter", "the halting problem", "RISC-V architecture", "mRNA vaccines",
        ],
    },
    {
        "name": "long_context_90k",
        "description": "~90k token input context, short generation — tests KV cache near limit",
        "max_tokens": 512,
        "system": "You are a code reviewer. Be concise.",
        "user": (
            "Review the following codebase for critical bugs and security issues. "
            "Output only the top 5 issues, one line each.\n\n"
            + "\n".join(
                f"# module_{m}/service.py\n"
                + "\n".join(
                    f"class Handler{m}_{c}:\n"
                    f"    \"\"\"Handles requests for module {m}, component {c}.\"\"\"\n"
                    f"    def __init__(self, db_url: str, api_key: str, timeout: int = 30):\n"
                    f"        self.db_url = db_url\n"
                    f"        self.api_key = api_key\n"
                    f"        self.timeout = timeout\n"
                    f"        self._cache: dict[str, Any] = {{}}\n"
                    f"        self._retry_count = 0\n"
                    f"\n"
                    f"    def process(self, request_id: str, payload: dict) -> dict:\n"
                    f"        if not request_id:\n"
                    f"            raise ValueError('request_id is required')\n"
                    f"        result = self._validate(payload)\n"
                    f"        transformed = self._transform(result, config_version={m * 10 + c})\n"
                    f"        self._cache[request_id] = transformed\n"
                    f"        return {{'status': 'ok', 'data': transformed, 'handler': '{m}_{c}'}}\n"
                    f"\n"
                    f"    def _validate(self, payload: dict) -> dict:\n"
                    f"        required = ['user_id', 'action', 'timestamp']\n"
                    f"        for field in required:\n"
                    f"            if field not in payload:\n"
                    f"                raise ValueError(f'Missing required field: {{field}}')\n"
                    f"        return {{k: v for k, v in payload.items() if v is not None}}\n"
                    f"\n"
                    f"    def _transform(self, data: dict, config_version: int) -> dict:\n"
                    f"        output = {{}}\n"
                    f"        for key, value in data.items():\n"
                    f"            if isinstance(value, str):\n"
                    f"                output[key] = value.strip().lower()\n"
                    f"            elif isinstance(value, (int, float)):\n"
                    f"                output[key] = value * (1 + config_version / 1000)\n"
                    f"            else:\n"
                    f"                output[key] = value\n"
                    f"        output['_version'] = config_version\n"
                    f"        return output\n"
                    for c in range(20)
                )
                for m in range(15)
            )
        ),
    },
    {
        "name": "long_context_gen_4k",
        "description": "~80k token input + 4096 token generation — sustained output near context limit",
        "max_tokens": 4096,
        "system": "You are an expert code auditor writing a detailed security report.",
        "user": (
            "Perform a comprehensive security audit of the following codebase. "
            "For each module, identify vulnerabilities, rate their severity (Critical/High/Medium/Low), "
            "provide a fix, and explain the attack vector. Write the full report.\n\n"
            + "\n".join(
                f"# service_{m}/handler.py\n"
                + "\n".join(
                    f"def handle_{m}_{f}(request: dict, db: 'Connection', config: dict) -> dict:\n"
                    f"    user_id = request.get('user_id', 'anonymous')\n"
                    f"    query = f\"SELECT * FROM users WHERE id = '{{user_id}}'\"\n"
                    f"    result = db.execute(query)\n"
                    f"    if not result:\n"
                    f"        return {{'error': 'not_found', 'handler': '{m}_{f}'}}\n"
                    f"    token = hashlib.md5(f'{{user_id}}:{{config[\"secret\"]}}'.encode()).hexdigest()\n"
                    f"    return {{'data': result, 'token': token, 'cache_key': f'user_{{user_id}}_{m}_{f}'}}\n"
                    f"\n"
                    for f in range(25)
                )
                for m in range(12)
            )
        ),
    },
    {
        "name": "parallel_2_slots",
        "description": "2 concurrent requests (max_num_seqs=2)",
        "max_tokens": 1024,
        "parallel": 2,
        "prompts": [
            {
                "system": "You are an expert Python developer.",
                "user": "Write a complete async HTTP client library with connection pooling, retry logic, and rate limiting. Include type hints and docstrings.",
            },
            {
                "system": "You are a thoughtful essayist.",
                "user": "Write a detailed essay on the future of renewable energy, covering solar, wind, hydrogen, and nuclear fusion. Discuss costs, scalability, and policy implications.",
            },
        ],
    },
    {
        "name": "parallel_3_slots",
        "description": "3 concurrent requests (tests beyond default max_num_seqs=2)",
        "max_tokens": 512,
        "parallel": 3,
        "prompts": [
            {
                "system": "You are a concise technical writer.",
                "user": "Explain how B-trees work and why they're used in databases. Include pseudocode for insert and search.",
            },
            {
                "system": "You are a concise technical writer.",
                "user": "Explain how consistent hashing works and why it's used in distributed systems. Include pseudocode.",
            },
            {
                "system": "You are a concise technical writer.",
                "user": "Explain how the Raft consensus algorithm works. Include pseudocode for leader election and log replication.",
            },
        ],
    },
]


def _run_single_stream(
    messages: list[dict], max_tokens: int, dry: bool, api_key: str,
) -> dict[str, Any]:
    """Run a single streaming request. Thread-safe (no shared state)."""
    spec_before = _scrape_raw_spec_counters()
    body: dict[str, Any] = {
        "model": "qwen3.6-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if dry:
        body.update(DRY_ON_PARAMS)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t_start = time.monotonic()
    t_first: float | None = None
    t_last = t_start
    prompt_tokens = completion_tokens = chunks = 0

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
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content") or delta.get("tool_calls"):
                        now = time.monotonic()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        chunks += 1

    total_time = t_last - t_start
    ttft = (t_first - t_start) if t_first else total_time
    decode_time = (t_last - t_first) if t_first else 0.001

    result = {
        "ttft_s": round(ttft, 3),
        "decode_tps": round(completion_tokens / decode_time, 2) if decode_time > 0 else 0,
        "overall_tps": round(completion_tokens / total_time, 2) if total_time > 0 else 0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": round(total_time, 3),
        "chunks": chunks,
    }
    time.sleep(1.0)  # let vLLM flush Prometheus counters
    spec_after = _scrape_raw_spec_counters()
    spec_m = compute_spec_metrics(spec_before, spec_after)
    if spec_m:
        result["spec_metrics"] = spec_m
    return result


def _run_parallel_requests(
    prompts: list[dict], max_tokens: int, dry: bool, api_key: str,
) -> dict[str, Any]:
    """Fire N requests concurrently, return aggregate metrics."""
    messages_list = [
        [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]
        for p in prompts
    ]
    n = len(messages_list)
    wall_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(_run_single_stream, msgs, max_tokens, dry, api_key)
            for msgs in messages_list
        ]
        results = []
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    wall_time = time.monotonic() - wall_start
    total_tokens = sum(r["completion_tokens"] for r in results)
    per_req_tps = [r["overall_tps"] for r in results]
    per_req_ttft = [r["ttft_s"] for r in results]

    return {
        "parallel_n": n,
        "wall_time_s": round(wall_time, 3),
        "aggregate_tps": round(total_tokens / wall_time, 2) if wall_time > 0 else 0,
        "per_request_tps_mean": round(statistics.mean(per_req_tps), 2),
        "per_request_tps_min": round(min(per_req_tps), 2),
        "per_request_tps_max": round(max(per_req_tps), 2),
        "per_request_ttft_mean": round(statistics.mean(per_req_ttft), 3),
        "total_completion_tokens": total_tokens,
        "individual_results": results,
    }


def run_phase_5(
    results_path: Path, completed: set[tuple], api_key: str,
    spec_winner: str, kv_winner: str, dry_winner: bool,
) -> None:
    """Phase 5: Stress test the optimal config with sustained, burst, and parallel workloads."""
    _log("\n=== Phase 5: Stress Test ===")
    _log(f"Config: kv={kv_winner} spec={spec_winner or '(none)'} dry={dry_winner}")

    restart_server(kv_winner, spec_winner)
    wait_for_health()
    # Warmup
    run_scenario(SCENARIOS[0], dry=False, api_key=api_key)

    # Count total tracker steps
    total_steps = 0
    for s in STRESS_SCENARIOS:
        if s.get("parallel"):
            total_steps += 3  # 3 runs, 1 step each
        elif s.get("burst"):
            total_steps += s["burst"] * 3
        else:
            total_steps += 3
    tracker = ProgressTracker(total_steps, "Phase 5: Stress")

    for stress in STRESS_SCENARIOS:
        is_parallel = stress.get("parallel", 0) > 0
        is_burst = stress.get("burst", 0) > 0

        for run_num in range(1, 4):  # 3 runs

            # --- Parallel slot test ---
            if is_parallel:
                key = (5, kv_winner, spec_winner, dry_winner,
                       stress["name"], run_num)
                if key in completed:
                    tracker.update(f"{stress['name']} run {run_num} (cached)")
                    continue

                run_start = time.monotonic()
                try:
                    spec_before_par = _scrape_raw_spec_counters()
                    metrics = _run_parallel_requests(
                        stress["prompts"], stress["max_tokens"],
                        dry_winner, api_key,
                    )
                    time.sleep(1.0)  # let vLLM flush Prometheus counters
                    spec_after_par = _scrape_raw_spec_counters()
                    spec_m = compute_spec_metrics(spec_before_par, spec_after_par)
                    if spec_m:
                        metrics["spec_metrics"] = spec_m
                    status = "ok"
                    error = None
                    _log(f"  {stress['name']} run {run_num}: "
                         f"aggregate={metrics['aggregate_tps']:.1f} tok/s, "
                         f"per_req_mean={metrics['per_request_tps_mean']:.1f} tok/s, "
                         f"wall={metrics['wall_time_s']:.1f}s")
                except Exception as e:
                    _log(f"\nERROR: {stress['name']} run {run_num}: {e}")
                    metrics = {}
                    status = "error"
                    error = str(e)

                result = {
                    "phase": 5,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_cache_dtype": kv_winner,
                        "speculative_config": spec_winner,
                        "config_label": f"stress_{stress['name']}",
                    },
                    "request_config": {"dry": dry_winner},
                    "scenario": stress["name"],
                    "run": run_num,
                    "stress_max_tokens": stress["max_tokens"],
                    "parallel_n": stress["parallel"],
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                }
                flush_result(result, results_path)

                run_duration = time.monotonic() - run_start
                tps = metrics.get("aggregate_tps") if metrics else None
                tracker.update(
                    f"{stress['name']} run {run_num}",
                    last_tps=tps, duration=run_duration,
                )
                continue

            # --- Burst and sequential tests ---
            iterations = stress.get("burst", 1) if is_burst else 1
            burst_results: list[dict[str, Any]] = []

            for burst_i in range(iterations):
                key = (5, kv_winner, spec_winner, dry_winner,
                       stress["name"], run_num * 100 + burst_i)
                if key in completed:
                    tracker.update(f"{stress['name']} run {run_num} (cached)")
                    continue

                user_msg = stress["user"]
                if is_burst and stress.get("burst_topics"):
                    user_msg = user_msg.format(topic=stress["burst_topics"][burst_i])

                messages = [
                    {"role": "system", "content": stress["system"]},
                    {"role": "user", "content": user_msg},
                ]

                run_start = time.monotonic()
                try:
                    metrics = _run_single_stream(
                        messages, stress["max_tokens"], dry_winner, api_key,
                    )
                    status = "ok"
                    error = None
                    burst_results.append(metrics)
                except Exception as e:
                    _log(f"\nERROR: {stress['name']} run {run_num}: {e}")
                    metrics = {}
                    status = "error"
                    error = str(e)

                result = {
                    "phase": 5,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "server_config": {
                        "kv_cache_dtype": kv_winner,
                        "speculative_config": spec_winner,
                        "config_label": f"stress_{stress['name']}",
                    },
                    "request_config": {"dry": dry_winner},
                    "scenario": stress["name"],
                    "run": run_num,
                    "burst_index": burst_i if is_burst else None,
                    "stress_max_tokens": stress["max_tokens"],
                    "metrics": metrics,
                    "status": status,
                    "error": error,
                }
                flush_result(result, results_path)

                run_duration = time.monotonic() - run_start
                tps = metrics.get("overall_tps") if metrics else None
                tracker.update(
                    f"{stress['name']} run {run_num}" + (f" burst {burst_i+1}/{iterations}" if is_burst else ""),
                    last_tps=tps, duration=run_duration,
                )

            if is_burst and burst_results:
                tps_vals = [m["overall_tps"] for m in burst_results]
                _log(f"  {stress['name']} run {run_num}: {len(burst_results)} bursts, "
                     f"mean={statistics.mean(tps_vals):.1f} min={min(tps_vals):.1f} max={max(tps_vals):.1f} tok/s")

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
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Run only this phase")
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

    dry_winner = False  # default

    if args.phase is None or args.phase == 3:
        run_phase_3(args.runs, results_path, completed, api_key, spec_winner, kv_winner)

    # Determine DRY winner from Phase 3 results
    p3 = [r for r in load_results(results_path) if r.get("phase") == 3 and r.get("status") == "ok"]
    if p3:
        dry_off_tps = [r["metrics"]["overall_tps"] for r in p3 if not r["request_config"]["dry"]]
        dry_on_tps = [r["metrics"]["overall_tps"] for r in p3 if r["request_config"]["dry"]]
        if dry_off_tps and dry_on_tps:
            dry_winner = statistics.mean(dry_on_tps) >= statistics.mean(dry_off_tps)
            _log(f"DRY winner: {'on' if dry_winner else 'off'}")

    if args.phase is None or args.phase == 4:
        run_phase_4(results_path, completed, api_key, spec_winner, kv_winner)

    if args.phase is None or args.phase == 5:
        run_phase_5(results_path, completed, api_key, spec_winner, kv_winner, dry_winner)

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
