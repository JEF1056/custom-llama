#!/usr/bin/env python3
"""Generate a Markdown benchmark report from JSONL results.

Called automatically by benchmark.py or standalone:
  python scripts/benchmark_report.py benchmark/results/20260603_142200_runs.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def _ok_results(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("status") == "ok"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(m, 1), round(s, 1)


def _pct_change(baseline: float, variant: float) -> str:
    if baseline == 0:
        return "N/A"
    diff = ((variant - baseline) / baseline) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}%"


def _spec_label(spec: str) -> str:
    if not spec:
        return "none"
    try:
        d = json.loads(spec)
        method = d.get("method", "?")
        if d.get("ngram_first"):
            return "ngram+mtp"
        return method
    except (json.JSONDecodeError, TypeError):
        return str(spec)[:15]


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_executive_summary(results: list[dict]) -> str:
    """Top configs and recommendation."""
    ok = _ok_results(results)
    if not ok:
        return "## Executive Summary\n\nNo successful benchmark results.\n"

    combo_tps: dict[str, list[float]] = {}
    for r in ok:
        sc = r["server_config"]
        rc = r["request_config"]
        key = f"kv={sc['kv_cache_dtype']} | spec={_spec_label(sc['speculative_config'])} | dry={'on' if rc['dry'] else 'off'}"
        combo_tps.setdefault(key, []).append(r["metrics"]["overall_tps"])

    ranked = sorted(combo_tps.items(), key=lambda x: statistics.mean(x[1]), reverse=True)

    lines = ["## Executive Summary\n"]
    lines.append("| Rank | Configuration | Mean tok/s | Std |")
    lines.append("|------|---------------|-----------|-----|")
    for i, (key, tps_list) in enumerate(ranked[:5], 1):
        m, s = _mean_std(tps_list)
        marker = " **<-- RECOMMENDED**" if i == 1 else ""
        lines.append(f"| {i} | {key} | {m} | +/-{s} |{marker}")

    # Recommended .env.default
    if ranked:
        best_key = ranked[0][0]
        lines.append(f"\n**Best overall configuration:** `{best_key}`\n")

    return "\n".join(lines) + "\n"


def _section_methodology() -> str:
    return """## Methodology

- **Hardware**: RTX 3090 24GB, single GPU
- **Model**: Qwen3.6-27B AutoRound INT4 (~19GB VRAM)
- **Engine**: vLLM 0.22.0 (turboquant fork)
- **Measurement**: Streaming chat completions, `max_tokens=1024`, `temperature=0`, thinking disabled
- **Metrics**:
  - **TTFT**: Time to first token (seconds)
  - **Decode tok/s**: `completion_tokens / (t_last_token - t_first_token)`
  - **Overall tok/s**: `completion_tokens / (t_last_token - t_start)`
  - **Spec acceptance rate**: from vLLM Prometheus `/metrics` endpoint (speculative configs only)
- **Phases**: Speculative decoding (Phase 1) → KV cache dtype (Phase 2) → DRY sampling (Phase 3) → Cross-validation (Phase 4)
- Each phase isolates one variable while holding others constant at the previous phase's winner.
"""


def _section_phase(
    results: list[dict],
    phase: int,
    title: str,
    config_field: str,
    label_fn: Any,
    baseline_value: str | None = None,
) -> str:
    """Generic phase results table."""
    ok = [r for r in _ok_results(results) if r.get("phase") == phase]
    if not ok:
        return f"## {title}\n\nNo results for this phase.\n"

    scenarios = sorted(set(r["scenario"] for r in ok))
    configs = sorted(set(label_fn(r["server_config"][config_field]) for r in ok))

    # Build data: config → scenario → [tps]
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ttft_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ok:
        cfg_label = label_fn(r["server_config"][config_field])
        data[cfg_label][r["scenario"]].append(r["metrics"]["overall_tps"])
        ttft_data[cfg_label][r["scenario"]].append(r["metrics"]["ttft_s"])

    # Overall tok/s table
    lines = [f"## {title}\n"]
    lines.append("### Overall Tokens/Second\n")

    header = "| Config | " + " | ".join(scenarios) + " | **Mean** |"
    sep = "|--------|" + "|".join(["------"] * len(scenarios)) + "|--------|"
    lines.append(header)
    lines.append(sep)

    baseline_means: dict[str, float] = {}
    config_overall: dict[str, float] = {}

    for cfg in configs:
        row = [cfg]
        all_tps = []
        for sc in scenarios:
            m, s = _mean_std(data[cfg].get(sc, []))
            row.append(f"{m} +/-{s}")
            all_tps.append(m)
        overall = round(statistics.mean(all_tps), 1) if all_tps else 0
        config_overall[cfg] = overall
        row.append(f"**{overall}**")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    # % change from baseline
    if baseline_value is not None:
        bl_label = label_fn(baseline_value)
        if bl_label in config_overall:
            lines.append("\n### % Change from Baseline\n")
            bl_overall = config_overall[bl_label]
            for cfg in configs:
                if cfg == bl_label:
                    continue
                pct = _pct_change(bl_overall, config_overall[cfg])
                lines.append(f"- **{cfg}** vs {bl_label}: {pct}")
            lines.append("")

    # TTFT table
    lines.append("\n### Time to First Token (seconds)\n")
    header = "| Config | " + " | ".join(scenarios) + " |"
    sep = "|--------|" + "|".join(["------"] * len(scenarios)) + "|"
    lines.append(header)
    lines.append(sep)
    for cfg in configs:
        row = [cfg]
        for sc in scenarios:
            m, s = _mean_std(ttft_data[cfg].get(sc, []))
            row.append(f"{m} +/-{s}")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    return "\n".join(lines) + "\n"


def _section_spec_acceptance(results: list[dict]) -> str:
    """Speculative decoding acceptance rate table from Prometheus metrics."""
    ok = [r for r in _ok_results(results) if r.get("phase") == 1]
    if not ok:
        return ""

    # Only include configs that have spec_metrics
    has_spec = [r for r in ok if r.get("metrics", {}).get("spec_metrics")]
    if not has_spec:
        return "### Speculative Acceptance Rate\n\nNo acceptance rate data available (metrics endpoint may not expose spec_decode counters for this vLLM version).\n"

    scenarios = sorted(set(r["scenario"] for r in has_spec))
    configs = sorted(set(r["server_config"].get("config_label", "?") for r in has_spec))

    # config → scenario → [acceptance_rate]
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in has_spec:
        label = r["server_config"].get("config_label", "?")
        ar = r["metrics"]["spec_metrics"].get("spec_acceptance_rate")
        if ar is not None:
            data[label][r["scenario"]].append(ar)

    if not any(data.values()):
        return ""

    lines = ["### Speculative Acceptance Rate\n"]
    header = "| Config | " + " | ".join(scenarios) + " | **Mean** |"
    sep = "|--------|" + "|".join(["------"] * len(scenarios)) + "|--------|"
    lines.append(header)
    lines.append(sep)

    for cfg in configs:
        row = [cfg]
        all_rates = []
        for sc in scenarios:
            vals = data[cfg].get(sc, [])
            if vals:
                m = round(statistics.mean(vals) * 100, 1)
                row.append(f"{m}%")
                all_rates.append(m)
            else:
                row.append("—")
        if all_rates:
            row.append(f"**{round(statistics.mean(all_rates), 1)}%**")
        else:
            row.append("—")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    lines.append("\n_Acceptance rate = fraction of draft tokens accepted by the verifier. Higher is better._\n")
    return "\n".join(lines) + "\n"


def _section_dry(results: list[dict]) -> str:
    """Phase 3: DRY on vs off comparison."""
    ok = [r for r in _ok_results(results) if r.get("phase") == 3]
    if not ok:
        return "## Phase 3: DRY Sampling Impact\n\nNo results for this phase.\n"

    scenarios = sorted(set(r["scenario"] for r in ok))

    dry_off: dict[str, list[float]] = defaultdict(list)
    dry_on: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        bucket = dry_on if r["request_config"]["dry"] else dry_off
        bucket[r["scenario"]].append(r["metrics"]["overall_tps"])

    lines = ["## Phase 3: DRY Sampling Impact\n"]
    lines.append("| Scenario | DRY Off (tok/s) | DRY On (tok/s) | Throughput Cost |")
    lines.append("|----------|----------------|----------------|-----------------|")

    off_total, on_total = [], []
    for sc in scenarios:
        m_off, s_off = _mean_std(dry_off.get(sc, []))
        m_on, s_on = _mean_std(dry_on.get(sc, []))
        pct = _pct_change(m_off, m_on)
        lines.append(f"| {sc} | {m_off} +/-{s_off} | {m_on} +/-{s_on} | {pct} |")
        off_total.append(m_off)
        on_total.append(m_on)

    if off_total and on_total:
        mean_off = round(statistics.mean(off_total), 1)
        mean_on = round(statistics.mean(on_total), 1)
        overall_pct = _pct_change(mean_off, mean_on)
        lines.append(f"| **Mean** | **{mean_off}** | **{mean_on}** | **{overall_pct}** |")

    lines.append("\n**Verdict**: " + (
        "DRY has negligible throughput impact."
        if off_total and on_total and abs(statistics.mean(on_total) - statistics.mean(off_total)) / max(statistics.mean(off_total), 0.01) < 0.03
        else "DRY has measurable throughput cost — consider whether output quality improvement justifies it."
    ))

    return "\n".join(lines) + "\n"


def _section_crossval(results: list[dict]) -> str:
    """Phase 4: Cross-validation results."""
    ok = [r for r in _ok_results(results) if r.get("phase") == 4]
    if not ok:
        return "## Phase 4: Cross-validation\n\nNo results for this phase.\n"

    scenarios = sorted(set(r["scenario"] for r in ok))

    combo_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ok:
        sc = r["server_config"]
        rc = r["request_config"]
        key = f"kv={sc['kv_cache_dtype'][:12]} spec={_spec_label(sc['speculative_config'])} dry={'on' if rc['dry'] else 'off'}"
        combo_data[key][r["scenario"]].append(r["metrics"]["overall_tps"])

    lines = ["## Phase 4: Cross-validation (5 runs)\n"]
    header = "| Configuration | " + " | ".join(scenarios) + " | **Mean** |"
    sep = "|---------------|" + "|".join(["------"] * len(scenarios)) + "|--------|"
    lines.append(header)
    lines.append(sep)

    for combo, scenario_data in sorted(combo_data.items()):
        row = [combo]
        all_means = []
        for sc in scenarios:
            m, s = _mean_std(scenario_data.get(sc, []))
            row.append(f"{m} +/-{s}")
            all_means.append(m)
        overall = round(statistics.mean(all_means), 1) if all_means else 0
        row.append(f"**{overall}**")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    return "\n".join(lines) + "\n"


def _section_best_per_scenario(results: list[dict]) -> str:
    """Best config per scenario type."""
    ok = _ok_results(results)
    if not ok:
        return ""

    scenarios = sorted(set(r["scenario"] for r in ok))

    lines = ["## Best Configuration per Scenario\n"]
    lines.append("| Scenario | Best Config | tok/s |")
    lines.append("|----------|-------------|-------|")

    for sc in scenarios:
        sc_results = [r for r in ok if r["scenario"] == sc]
        combo_tps: dict[str, list[float]] = {}
        for r in sc_results:
            s = r["server_config"]
            rc = r["request_config"]
            key = f"kv={s['kv_cache_dtype']} spec={_spec_label(s['speculative_config'])} dry={'on' if rc['dry'] else 'off'}"
            combo_tps.setdefault(key, []).append(r["metrics"]["overall_tps"])

        if combo_tps:
            best = max(combo_tps, key=lambda k: statistics.mean(combo_tps[k]))
            m = round(statistics.mean(combo_tps[best]), 1)
            lines.append(f"| {sc} | {best} | {m} |")

    return "\n".join(lines) + "\n"


def _section_recommended_env(results: list[dict]) -> str:
    """Copy-pasteable .env.default recommendation."""
    ok = _ok_results(results)
    if not ok:
        return ""

    combo_tps: dict[tuple, list[float]] = {}
    for r in ok:
        sc = r["server_config"]
        rc = r["request_config"]
        combo = (sc["kv_cache_dtype"], sc["speculative_config"], rc["dry"])
        combo_tps.setdefault(combo, []).append(r["metrics"]["overall_tps"])

    if not combo_tps:
        return ""

    best_combo = max(combo_tps, key=lambda k: statistics.mean(combo_tps[k]))
    kv, spec, dry = best_combo

    dry_line = (
        'LLM_DRY_CONFIG={"multiplier": 0.4, "base": 1.75, "allowed_length": 128, "penalty_last_n": 2048}'
        if dry
        else "LLM_DRY_CONFIG="
    )
    spec_line = f"LLM_SPECULATIVE_CONFIG={spec}" if spec else "LLM_SPECULATIVE_CONFIG="

    return f"""## Recommended .env.default

Copy these values into your `.env.default`:

```bash
# KV cache quantization — benchmark winner
LLM_KV_CACHE_DTYPE={kv}

# Speculative decoding — benchmark winner
{spec_line}

# DRY sampling — benchmark winner
{dry_line}
```

Mean throughput: **{round(statistics.mean(combo_tps[best_combo]), 1)} tok/s** across all scenarios.
"""


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(results_path: Path, report_path: Path) -> None:
    """Generate full Markdown report from JSONL results."""
    results = load_results(results_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        f"# vLLM Benchmark Report\n\n_Generated {ts} from `{results_path.name}`_\n",
        _section_executive_summary(results),
        _section_methodology(),
        _section_phase(
            results, phase=1,
            title="Phase 1: Speculative Decoding Impact",
            config_field="config_label",
            label_fn=lambda x: x,
            baseline_value="none",
        ),
        _section_spec_acceptance(results),
        _section_phase(
            results, phase=2,
            title="Phase 2: KV Cache Dtype Impact",
            config_field="kv_cache_dtype",
            label_fn=lambda x: x,
            baseline_value="rotorquant_k4v2_nc",
        ),
        _section_dry(results),
        _section_crossval(results),
        _section_best_per_scenario(results),
        _section_recommended_env(results),
        f"\n---\n\n_Raw data: `{results_path.name}`_\n",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_report.py <results.jsonl> [output.md]")
        sys.exit(1)
    results_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        report_path = Path(sys.argv[2])
    else:
        report_path = results_path.with_name(
            results_path.stem.replace("_runs", "_report") + ".md"
        )
    generate_report(results_path, report_path)
    print(f"Report written to {report_path}")
