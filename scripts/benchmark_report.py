#!/usr/bin/env python3
"""Generate a Markdown benchmark report from JSONL results.

Called automatically by benchmark.py or standalone:
  python scripts/benchmark_report.py benchmark/results/20260603_142200_runs.jsonl
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# KV cache bit-sum: lower = more compressed = preferred as tiebreaker
# ---------------------------------------------------------------------------

def _kv_bit_sum(kv_dtype: str) -> int:
    """Extract bit-sum from KV cache dtype name. Lower = more compressed."""
    name = kv_dtype.lower()
    m = re.search(r'k(\d+)v(\d+)', name)
    if m:
        return int(m.group(1)) + int(m.group(2))
    m = re.search(r'(\d+)bit', name)
    if m:
        return int(m.group(1)) * 2  # 3bit ≈ k3v3 = 6
    return 99  # unknown, rank last


# ---------------------------------------------------------------------------
# DRY config for recommendation output
# ---------------------------------------------------------------------------

DRY_ON_PARAMS = {
    "multiplier": 0.4,
    "base": 1.75,
    "allowed_length": 128,
    "penalty_last_n": 2048,
}


def load_results(path: Path) -> list[dict[str, Any]]:
    """Lines starting with # are treated as comments and skipped."""
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
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
        parts = [method]
        n = d.get("num_speculative_tokens")
        if n is not None:
            parts.append(f"n{n}")
        if d.get("ngram_fallback"):
            parts.append("nfb")
        if d.get("draft_sample_method") == "probabilistic":
            parts.append("prob")
        return "_".join(parts)
    except (json.JSONDecodeError, TypeError):
        return str(spec)[:15]


# ---------------------------------------------------------------------------
# Phase winner derivation (with tiebreakers)
# ---------------------------------------------------------------------------

def _pick_best_config(phase_results: list[dict], config_key: str) -> str:
    """Pick config with highest mean tps from phase results."""
    by_config: dict[str, list[float]] = {}
    for r in phase_results:
        val = r["server_config"][config_key]
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        by_config.setdefault(val, []).append(r["metrics"].get(tps_key, 0))
    if not by_config:
        return ""
    return max(by_config, key=lambda k: statistics.mean(by_config[k]))


def _pick_best_kv(phase_results: list[dict]) -> str:
    """Pick KV config: highest mean tps, prefer lower bit-sum within 3%."""
    by_config: dict[str, list[float]] = {}
    for r in phase_results:
        val = r["server_config"]["kv_cache_dtype"]
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        by_config.setdefault(val, []).append(r["metrics"].get(tps_key, 0))
    if not by_config:
        return ""
    ranked = sorted(by_config.items(), key=lambda x: statistics.mean(x[1]), reverse=True)
    top_mean = statistics.mean(ranked[0][1])
    # All configs within 3% of top are candidates
    candidates = [(val, tps) for val, tps in ranked
                  if statistics.mean(tps) >= top_mean * 0.97]
    # Among candidates, prefer lowest bit-sum
    return min(candidates, key=lambda x: _kv_bit_sum(x[0]))[0]


def _pick_dry_winner(phase_results: list[dict]) -> bool:
    """DRY on wins if within 3% of DRY off throughput (quality benefit)."""
    dry_off_tps: list[float] = []
    dry_on_tps: list[float] = []
    for r in phase_results:
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        tps = r["metrics"].get(tps_key, 0)
        if r["request_config"]["dry"]:
            dry_on_tps.append(tps)
        else:
            dry_off_tps.append(tps)
    if not dry_off_tps or not dry_on_tps:
        return False
    mean_off = statistics.mean(dry_off_tps)
    mean_on = statistics.mean(dry_on_tps)
    return mean_on >= mean_off * 0.97


def _derive_winners(results: list[dict]) -> tuple[str, str, bool]:
    """Derive (spec_winner, kv_winner, dry_winner) from individual phases.

    Applies KV bit-sum tiebreaker and DRY 3% preference.
    Falls back to all-phase data when a specific phase is missing.
    """
    ok = _ok_results(results)
    if not ok:
        return "", "", False

    p1 = [r for r in ok if r.get("phase") == 1]
    p2 = [r for r in ok if r.get("phase") == 2]
    p3 = [r for r in ok if r.get("phase") == 3]

    # Phase 1 → spec winner; fallback to all data
    spec_winner = (_pick_best_config(p1, "speculative_config") if p1
                   else _pick_best_config(ok, "speculative_config"))

    # Phase 2 → KV winner with bit-sum tiebreaker; fallback to all data
    kv_winner = _pick_best_kv(p2) if p2 else _pick_best_kv(ok)

    # Phase 3 → DRY winner with 3% threshold
    dry_winner = _pick_dry_winner(p3) if p3 else False

    return spec_winner, kv_winner, dry_winner


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_executive_summary(results: list[dict]) -> str:
    """Top configs and recommendation.

    Uses Phase 4 (cross-validation) data when available for the ranking table.
    Falls back to all-phase data otherwise.
    Derives the RECOMMENDED pick from individual phase winners with tiebreakers.
    """
    ok = _ok_results(results)
    if not ok:
        return "## Executive Summary\n\nNo successful benchmark results.\n"

    # Derive the recommended config from phase winners (with tiebreakers)
    spec_winner, kv_winner, dry_winner = _derive_winners(results)
    rec_key = (
        f"kv={kv_winner} | spec={_spec_label(spec_winner)} "
        f"| dry={'on' if dry_winner else 'off'}"
    )

    # Prefer Phase 4 (cross-val) data for ranking — it's the controlled comparison.
    p4 = [r for r in ok if r.get("phase") == 4]
    ranking_source = p4 if len(p4) >= 3 else ok
    source_label = "Phase 4 cross-validation" if ranking_source is p4 else "all phases"

    combo_tps: dict[str, list[float]] = {}
    for r in ranking_source:
        sc = r["server_config"]
        rc = r["request_config"]
        key = (
            f"kv={sc['kv_cache_dtype']} | spec={_spec_label(sc['speculative_config'])} "
            f"| dry={'on' if rc['dry'] else 'off'}"
        )
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        combo_tps.setdefault(key, []).append(r["metrics"][tps_key])

    ranked = sorted(combo_tps.items(), key=lambda x: statistics.mean(x[1]), reverse=True)

    lines = ["## Executive Summary\n"]
    lines.append(f"_Ranking from {source_label}. "
                 "tool_calling uses decode_tps (strips TTFT) for stable comparison._\n")
    lines.append("| Rank | Configuration | Mean tok/s | Std |")
    lines.append("|------|---------------|-----------|-----|")
    for i, (key, tps_list) in enumerate(ranked[:5], 1):
        m, s = _mean_std(tps_list)
        marker = " **<-- RECOMMENDED**" if key == rec_key else ""
        lines.append(f"| {i} | {key} | {m} | +/-{s} |{marker}")

    lines.append(f"\n**Recommended configuration:** `{rec_key}`")
    notes = []
    if dry_winner:
        notes.append("DRY on: throughput within 3% of DRY off — quality benefit outweighs cost")
    if kv_winner:
        notes.append(f"KV {kv_winner}: bit-sum={_kv_bit_sum(kv_winner)} "
                     "(lower = more compressed, preferred when performance is equivalent)")
    if notes:
        for note in notes:
            lines.append(f"- _{note}_")
    lines.append("")

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
    # Use decode_tps for tool_calling (output length varies, overall_tps is TTFT-dominated)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ttft_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ok:
        cfg_label = label_fn(r["server_config"][config_field])
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        data[cfg_label][r["scenario"]].append(r["metrics"][tps_key])
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
            vals = data[cfg].get(sc, [])
            if vals:
                m, s = _mean_std(vals)
                row.append(f"{m} +/-{s}")
                all_tps.append(m)
            else:
                row.append("—")
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
            vals = ttft_data[cfg].get(sc, [])
            if vals:
                m, s = _mean_std(vals)
                row.append(f"{m} +/-{s}")
            else:
                row.append("—")
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

    # config → scenario → [acceptance_pct] (already 0-100 from compute_spec_metrics)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in has_spec:
        label = r["server_config"].get("config_label", "?")
        sm = r["metrics"]["spec_metrics"]
        ar = sm.get("acceptance_pct")
        if ar is not None:
            data[label][r["scenario"]].append(ar)

    if not any(data.values()):
        return ""

    lines = ["### Speculative Acceptance Rate (per-run)\n"]
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
                m = round(statistics.mean(vals), 1)
                row.append(f"{m}%")
                all_rates.append(m)
            else:
                row.append("—")
        if all_rates:
            row.append(f"**{round(statistics.mean(all_rates), 1)}%**")
        else:
            row.append("—")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    # Also show drafted/accepted token counts
    lines.append("")
    lines.append("#### Draft Token Efficiency\n")
    lines.append("| Config | Drafted/run | Accepted/run | Efficiency |")
    lines.append("|--------|-----------|-------------|------------|")
    for cfg in configs:
        cfg_results = [r for r in has_spec if r["server_config"].get("config_label") == cfg]
        drafted = [r["metrics"]["spec_metrics"].get("drafted_tokens", 0) for r in cfg_results]
        accepted = [r["metrics"]["spec_metrics"].get("accepted_tokens", 0) for r in cfg_results]
        if drafted and any(d > 0 for d in drafted):
            m_d = round(statistics.mean(drafted))
            m_a = round(statistics.mean(accepted))
            eff = round((m_a / m_d) * 100, 1) if m_d > 0 else 0
            lines.append(f"| {cfg} | {m_d} | {m_a} | {eff}% |")

    lines.append("\n_Acceptance = drafted tokens accepted by verifier. Computed per-run from Prometheus counter deltas._\n")
    return "\n".join(lines) + "\n"


def _section_kv_token_efficiency(results: list[dict]) -> str:
    """Token efficiency table for Phase 2: tok/s per bit of KV cache."""
    ok = [r for r in _ok_results(results) if r.get("phase") == 2]
    if not ok:
        return ""

    scenarios = sorted(set(r["scenario"] for r in ok))
    configs = sorted(set(r["server_config"]["kv_cache_dtype"] for r in ok))

    # config → scenario → [tps]
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ok:
        kv = r["server_config"]["kv_cache_dtype"]
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        data[kv][r["scenario"]].append(r["metrics"].get(tps_key, 0))

    lines = ["### Token Efficiency (tok/s per KV bit)\n"]
    lines.append("_Higher = more throughput per unit of cache memory. "
                 "Computed as mean tok/s ÷ bit-sum (K bits + V bits)._\n")

    header = "| KV Dtype | Bit-sum | " + " | ".join(scenarios) + " | **Mean Efficiency** |"
    sep = "|----------|---------|" + "|".join(["------"] * len(scenarios)) + "|---------------------|"
    lines.append(header)
    lines.append(sep)

    efficiency_overall: dict[str, float] = {}
    for cfg in configs:
        bits = _kv_bit_sum(cfg)
        row = [cfg, str(bits)]
        all_eff = []
        for sc in scenarios:
            vals = data[cfg].get(sc, [])
            if vals:
                m = statistics.mean(vals)
                eff = round(m / bits, 2) if bits > 0 else 0
                row.append(f"{eff}")
                all_eff.append(eff)
            else:
                row.append("—")
        mean_eff = round(statistics.mean(all_eff), 2) if all_eff else 0
        efficiency_overall[cfg] = mean_eff
        row.append(f"**{mean_eff}**")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    # Highlight the most efficient config
    if efficiency_overall:
        best = max(efficiency_overall, key=efficiency_overall.get)  # type: ignore[arg-type]
        lines.append(f"\n**Most token-efficient**: `{best}` "
                     f"(bit-sum={_kv_bit_sum(best)}, "
                     f"efficiency={efficiency_overall[best]} tok/s/bit)")

    # MTP acceptance rate per KV config
    has_spec = [r for r in ok if r.get("metrics", {}).get("spec_metrics")]
    if has_spec:
        acc_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in has_spec:
            kv = r["server_config"]["kv_cache_dtype"]
            ar = r["metrics"]["spec_metrics"].get("acceptance_pct")
            if ar is not None:
                acc_data[kv][r["scenario"]].append(ar)

        if any(acc_data.values()):
            lines.append("")
            lines.append("### MTP Acceptance Rate by KV Dtype\n")
            lines.append("_Shows how KV cache quantization affects speculative decoding acceptance. "
                         "Lower acceptance → more wasted draft compute._\n")

            header = "| KV Dtype | Bit-sum | " + " | ".join(scenarios) + " | **Mean** |"
            sep = "|----------|---------|" + "|".join(["------"] * len(scenarios)) + "|--------|"
            lines.append(header)
            lines.append(sep)

            for cfg in configs:
                bits = _kv_bit_sum(cfg)
                row = [cfg, str(bits)]
                all_rates: list[float] = []
                for sc in scenarios:
                    vals = acc_data[cfg].get(sc, [])
                    if vals:
                        m = round(statistics.mean(vals), 1)
                        row.append(f"{m}%")
                        all_rates.append(m)
                    else:
                        row.append("—")
                if all_rates:
                    row.append(f"**{round(statistics.mean(all_rates), 1)}%**")
                else:
                    row.append("—")
                lines.append("| " + " | ".join(str(x) for x in row) + " |")

    lines.append("")
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
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        bucket[r["scenario"]].append(r["metrics"][tps_key])

    lines = ["## Phase 3: DRY Sampling Impact\n"]
    lines.append("| Scenario | DRY Off (tok/s) | DRY On (tok/s) | Throughput Cost |")
    lines.append("|----------|----------------|----------------|-----------------|")

    off_total, on_total = [], []
    for sc in scenarios:
        off_vals = dry_off.get(sc, [])
        on_vals = dry_on.get(sc, [])
        off_cell = f"{_mean_std(off_vals)[0]} +/-{_mean_std(off_vals)[1]}" if off_vals else "—"
        on_cell = f"{_mean_std(on_vals)[0]} +/-{_mean_std(on_vals)[1]}" if on_vals else "—"
        if off_vals and on_vals:
            pct = _pct_change(_mean_std(off_vals)[0], _mean_std(on_vals)[0])
            off_total.append(_mean_std(off_vals)[0])
            on_total.append(_mean_std(on_vals)[0])
        else:
            pct = "—"
        lines.append(f"| {sc} | {off_cell} | {on_cell} | {pct} |")

    if off_total and on_total:
        mean_off = round(statistics.mean(off_total), 1)
        mean_on = round(statistics.mean(on_total), 1)
        overall_pct = _pct_change(mean_off, mean_on)
        lines.append(f"| **Mean** | **{mean_off}** | **{mean_on}** | **{overall_pct}** |")

    if off_total and on_total:
        cost_pct = (statistics.mean(on_total) - statistics.mean(off_total)) / max(statistics.mean(off_total), 0.01)
        if abs(cost_pct) < 0.03:
            verdict = "DRY has negligible throughput impact (<3%). **Recommend DRY on** for output quality."
        else:
            verdict = "DRY has measurable throughput cost — consider whether output quality improvement justifies it."
    else:
        verdict = "Insufficient data to compare DRY on vs off."
    lines.append(f"\n**Verdict**: {verdict}")

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
        spec_raw = sc["speculative_config"]
        spec_detail = _spec_label(spec_raw)
        if spec_raw:
            # Include full JSON for unambiguous reproduction
            spec_detail += f" (`{spec_raw}`)"
        key = f"kv={sc['kv_cache_dtype']} | spec={spec_detail} | dry={'on' if rc['dry'] else 'off'}"
        tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
        combo_data[key][r["scenario"]].append(r["metrics"][tps_key])

    lines = ["## Phase 4: Cross-validation (5 runs)\n"]
    header = "| Configuration | " + " | ".join(scenarios) + " | **Mean** |"
    sep = "|---------------|" + "|".join(["------"] * len(scenarios)) + "|--------|"
    lines.append(header)
    lines.append(sep)

    for combo, scenario_data in sorted(combo_data.items()):
        row = [combo]
        all_means = []
        for sc in scenarios:
            vals = scenario_data.get(sc, [])
            if vals:
                m, s = _mean_std(vals)
                row.append(f"{m} +/-{s}")
                all_means.append(m)
            else:
                row.append("—")
        overall = round(statistics.mean(all_means), 1) if all_means else 0
        row.append(f"**{overall}**")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    return "\n".join(lines) + "\n"


def _section_stress_test(results: list[dict]) -> str:
    """Phase 5: Stress test results."""
    ok = [r for r in _ok_results(results) if r.get("phase") == 5]
    if not ok:
        return "## Phase 5: Stress Test\n\nNo stress test results.\n"

    scenarios = sorted(set(r["scenario"] for r in ok))

    lines = ["## Phase 5: Stress Test (Optimal Config)\n"]

    # Split parallel vs sequential results
    parallel_results = [r for r in ok if r.get("parallel_n")]
    sequential_results = [r for r in ok if not r.get("parallel_n")]
    seq_scenarios = sorted(set(r["scenario"] for r in sequential_results))

    # Sequential summary table
    if sequential_results:
        lines.append("### Sequential Tests\n")
        lines.append("| Test | Max Tokens | Runs | Mean tok/s | Min | Max | Mean TTFT |")
        lines.append("|------|-----------|------|-----------|-----|-----|-----------|")

        for sc_name in seq_scenarios:
            sc_results = [r for r in sequential_results if r["scenario"] == sc_name]
            tps_vals = [r["metrics"]["overall_tps"] for r in sc_results if r["metrics"].get("overall_tps")]
            ttft_vals = [r["metrics"]["ttft_s"] for r in sc_results if r["metrics"].get("ttft_s")]
            max_tok = sc_results[0].get("stress_max_tokens", "?") if sc_results else "?"

            if tps_vals:
                m_tps = round(statistics.mean(tps_vals), 1)
                min_tps = round(min(tps_vals), 1)
                max_tps = round(max(tps_vals), 1)
            else:
                m_tps = min_tps = max_tps = 0

            m_ttft = round(statistics.mean(ttft_vals), 2) if ttft_vals else 0

            lines.append(f"| {sc_name} | {max_tok} | {len(sc_results)} | {m_tps} | {min_tps} | {max_tps} | {m_ttft}s |")

    # Parallel slot analysis
    if parallel_results:
        lines.append("\n### Parallel Slot Performance\n")
        par_scenarios = sorted(set(r["scenario"] for r in parallel_results))

        lines.append("| Test | Slots | Runs | Aggregate tok/s | Per-req tok/s (mean) | Per-req tok/s (min) | Wall time |")
        lines.append("|------|-------|------|----------------|---------------------|--------------------:|-----------|")

        for sc_name in par_scenarios:
            sc_runs = [r for r in parallel_results if r["scenario"] == sc_name]
            n_slots = sc_runs[0].get("parallel_n", "?")
            agg_tps = [r["metrics"]["aggregate_tps"] for r in sc_runs if r["metrics"].get("aggregate_tps")]
            per_tps = [r["metrics"]["per_request_tps_mean"] for r in sc_runs if r["metrics"].get("per_request_tps_mean")]
            per_min = [r["metrics"]["per_request_tps_min"] for r in sc_runs if r["metrics"].get("per_request_tps_min")]
            wall = [r["metrics"]["wall_time_s"] for r in sc_runs if r["metrics"].get("wall_time_s")]

            m_agg = round(statistics.mean(agg_tps), 1) if agg_tps else 0
            m_per = round(statistics.mean(per_tps), 1) if per_tps else 0
            m_pmin = round(statistics.mean(per_min), 1) if per_min else 0
            m_wall = round(statistics.mean(wall), 1) if wall else 0

            lines.append(f"| {sc_name} | {n_slots} | {len(sc_runs)} | {m_agg} | {m_per} | {m_pmin} | {m_wall}s |")

        # Compare single vs parallel throughput
        # Find single-request baseline from sequential tests
        single_tps_vals = []
        for r in sequential_results:
            if r["metrics"].get("overall_tps") and r.get("stress_max_tokens") == 1024:
                single_tps_vals.append(r["metrics"]["overall_tps"])
        if not single_tps_vals:
            # Fall back to any sequential result
            single_tps_vals = [r["metrics"]["overall_tps"] for r in sequential_results if r["metrics"].get("overall_tps")]

        if single_tps_vals:
            single_baseline = statistics.mean(single_tps_vals)
            lines.append(f"\n**Single-request baseline**: {round(single_baseline, 1)} tok/s\n")
            for sc_name in par_scenarios:
                sc_runs = [r for r in parallel_results if r["scenario"] == sc_name]
                n_slots = sc_runs[0].get("parallel_n", "?")
                per_tps = [r["metrics"]["per_request_tps_mean"] for r in sc_runs if r["metrics"].get("per_request_tps_mean")]
                agg_tps = [r["metrics"]["aggregate_tps"] for r in sc_runs if r["metrics"].get("aggregate_tps")]
                if per_tps:
                    m_per = statistics.mean(per_tps)
                    m_agg = statistics.mean(agg_tps) if agg_tps else 0
                    degradation = _pct_change(single_baseline, m_per)
                    lines.append(f"- **{n_slots} slots**: per-request {degradation} vs single | aggregate {round(m_agg, 1)} tok/s total throughput")

    # Sustained generation analysis
    sustained = [r for r in ok if "sustained" in r["scenario"]]
    if sustained:
        lines.append("\n### Sustained Generation Analysis\n")
        for sc_name in sorted(set(r["scenario"] for r in sustained)):
            sc_runs = [r for r in sustained if r["scenario"] == sc_name]
            tps_vals = [r["metrics"]["overall_tps"] for r in sc_runs if r["metrics"].get("overall_tps")]
            tok_vals = [r["metrics"]["completion_tokens"] for r in sc_runs if r["metrics"].get("completion_tokens")]
            if tps_vals and tok_vals:
                m, s = _mean_std(tps_vals)
                avg_tok = round(statistics.mean(tok_vals))
                lines.append(f"- **{sc_name}**: {m} +/-{s} tok/s, avg {avg_tok} tokens generated")

    # Burst analysis
    burst = [r for r in ok if r.get("burst_index") is not None]
    if burst:
        lines.append("\n### Rapid Burst Analysis\n")
        by_run: dict[int, list[float]] = defaultdict(list)
        for r in burst:
            if r["metrics"].get("overall_tps"):
                by_run[r["run"]].append(r["metrics"]["overall_tps"])

        all_burst_tps = [r["metrics"]["overall_tps"] for r in burst if r["metrics"].get("overall_tps")]
        if all_burst_tps:
            lines.append(f"- **{len(all_burst_tps)} total burst requests** across {len(by_run)} runs")
            lines.append(f"- Mean: {round(statistics.mean(all_burst_tps), 1)} tok/s")
            lines.append(f"- Min: {round(min(all_burst_tps), 1)} tok/s | Max: {round(max(all_burst_tps), 1)} tok/s")
            if len(all_burst_tps) > 1:
                lines.append(f"- Stddev: {round(statistics.stdev(all_burst_tps), 1)} (lower = more consistent)")

    # Spec metrics if available
    spec_runs = [r for r in ok if r.get("metrics", {}).get("spec_metrics", {}).get("acceptance_pct")]
    if spec_runs:
        rates = [r["metrics"]["spec_metrics"]["acceptance_pct"] for r in spec_runs]
        lines.append(f"\n### Speculative Acceptance Under Stress\n")
        lines.append(f"- Mean acceptance rate: **{round(statistics.mean(rates), 1)}%**")
        lines.append(f"- Range: {round(min(rates), 1)}% — {round(max(rates), 1)}%")

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
            tps_key = "decode_tps" if sc == "tool_calling" else "overall_tps"
            combo_tps.setdefault(key, []).append(r["metrics"][tps_key])

        if combo_tps:
            best = max(combo_tps, key=lambda k: statistics.mean(combo_tps[k]))
            m = round(statistics.mean(combo_tps[best]), 1)
            suffix = " (decode)" if sc == "tool_calling" else ""
            lines.append(f"| {sc} | {best} | {m}{suffix} |")

    return "\n".join(lines) + "\n"


def _section_recommended_env(results: list[dict]) -> str:
    """Copy-pasteable .env.default recommendation.

    Derives winners from individual phases with tiebreakers:
    - KV: prefer lower bit-sum when configs are within 3%
    - DRY: prefer on when throughput cost is within 3% (quality benefit)
    """
    ok = _ok_results(results)
    if not ok:
        return ""

    spec_winner, kv_winner, dry_winner = _derive_winners(results)

    if not kv_winner and not spec_winner:
        return ""

    dry_line = (
        f'LLM_DRY_CONFIG={json.dumps(DRY_ON_PARAMS)}'
        if dry_winner
        else "LLM_DRY_CONFIG="
    )
    spec_line = (
        f"LLM_SPECULATIVE_CONFIG={spec_winner}"
        if spec_winner
        else "LLM_SPECULATIVE_CONFIG="
    )

    # Compute mean tps for the recommended combo from whatever phase data matches
    combo_tps: list[float] = []
    for r in ok:
        sc = r["server_config"]
        rc = r["request_config"]
        if (sc["kv_cache_dtype"] == kv_winner
                and sc["speculative_config"] == spec_winner
                and rc["dry"] == dry_winner):
            tps_key = "decode_tps" if r["scenario"] == "tool_calling" else "overall_tps"
            combo_tps.append(r["metrics"].get(tps_key, 0))

    tps_note = (
        f"Mean throughput: **{round(statistics.mean(combo_tps), 1)} tok/s** "
        "across matched runs."
        if combo_tps
        else ""
    )

    notes = []
    if dry_winner:
        notes.append("- **DRY on**: throughput within 3% of DRY off — "
                      "output quality benefit outweighs marginal cost")
    notes.append(f"- **KV {kv_winner}**: bit-sum = {_kv_bit_sum(kv_winner)} "
                 "(lower = more VRAM-efficient; preferred when performance is equivalent)")

    return f"""## Recommended .env.default

Copy these values into your `.env.default`:

```bash
# KV cache quantization — benchmark winner
LLM_KV_CACHE_DTYPE={kv_winner}

# Speculative decoding — benchmark winner
{spec_line}

# DRY sampling — benchmark winner
{dry_line}
```

{tps_note}

### Selection rationale

{chr(10).join(notes)}
"""


def _phase2_title(results: list[dict]) -> str:
    """Dynamic Phase 2 title showing the spec config held constant."""
    ok = [r for r in _ok_results(results) if r.get("phase") == 2]
    if ok:
        spec = ok[0]["server_config"].get("speculative_config", "")
        label = _spec_label(spec)
        return f"Phase 2: KV Cache Dtype Impact (spec={label})"
    # No Phase 2 data yet — derive from Phase 1 winner
    p1 = [r for r in _ok_results(results) if r.get("phase") == 1]
    if p1:
        winner = _pick_best_config(p1, "speculative_config")
        return f"Phase 2: KV Cache Dtype Impact (spec={_spec_label(winner)})"
    return "Phase 2: KV Cache Dtype Impact"


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
            title=_phase2_title(results),
            config_field="kv_cache_dtype",
            label_fn=lambda x: x,
            baseline_value="turboquant_k4v2_nc",
        ),
        _section_kv_token_efficiency(results),
        _section_dry(results),
        _section_crossval(results),
        _section_stress_test(results),
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
