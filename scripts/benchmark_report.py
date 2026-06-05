#!/usr/bin/env python3
"""Generate a Markdown benchmark report from JSONL results.

Called automatically by benchmark.py or standalone:
  python scripts/benchmark_report.py benchmark/results/20260604_142200_runs.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# KV cache bit-sum: lower = more compressed = preferred as tiebreaker
# ---------------------------------------------------------------------------

def _kv_bit_sum(kv_k: str, kv_v: str) -> int:
    """Approximate total bits from llama.cpp KV cache type names."""
    bits = {"f16": 16, "q8_0": 8, "q4_0": 4, "turbo4": 4, "turbo3": 3,
            "turbo2": 2, "turbo1": 1}
    return bits.get(kv_k, 16) + bits.get(kv_v, 16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase winner derivation
# ---------------------------------------------------------------------------

def _pick_phase_winner(
    results: list[dict], phase: int, group_key: str,
) -> tuple[str, float]:
    """Pick best config by mean overall_tps for a given phase."""
    phase_ok = [r for r in _ok_results(results) if r.get("phase") == phase]
    if not phase_ok:
        return ("", 0.0)

    groups: dict[str, list[float]] = {}
    for r in phase_ok:
        label = r["server_config"].get(group_key, r["server_config"].get("config_label", ""))
        tps = r["metrics"].get("overall_tps", 0)
        groups.setdefault(label, []).append(tps)

    ranked = sorted(groups.items(), key=lambda x: statistics.mean(x[1]), reverse=True)
    winner, tps_list = ranked[0]
    return (winner, statistics.mean(tps_list))


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_phase(
    results: list[dict], phase: int, title: str, group_key: str,
) -> list[str]:
    """Generate a table for a config sweep phase."""
    phase_results = [r for r in results if r.get("phase") == phase]
    ok = _ok_results(phase_results)
    errors = [r for r in phase_results if r.get("status") != "ok"]

    if not phase_results:
        return [f"## Phase {phase}: {title}", "", "*No data*", ""]

    lines = [f"## Phase {phase}: {title}", ""]

    # Group by config label
    groups: dict[str, list[dict]] = {}
    for r in ok:
        label = r["server_config"].get(group_key, r["server_config"].get("config_label", ""))
        groups.setdefault(label, []).append(r)

    if not groups:
        lines.append("*All runs failed.*")
        if errors:
            lines.append("")
            lines.append(f"Errors: {len(errors)}")
            for e in errors[:5]:
                lines.append(f"- `{e.get('scenario', '?')}`: {e.get('error', '?')[:100]}")
        lines.append("")
        return lines

    # Table
    lines.append("| Config | Decode tok/s | Overall tok/s | TTFT (s) | Avg Tokens | Runs |")
    lines.append("|--------|-------------|--------------|----------|------------|------|")

    ranked: list[tuple[str, float, str]] = []
    for label, runs in sorted(groups.items()):
        d_vals = [r["metrics"]["decode_tps"] for r in runs]
        o_vals = [r["metrics"]["overall_tps"] for r in runs]
        t_vals = [r["metrics"]["ttft_s"] for r in runs]
        tok_vals = [r["metrics"]["completion_tokens"] for r in runs]

        d_m, d_s = _mean_std(d_vals)
        o_m, o_s = _mean_std(o_vals)
        t_m, _ = _mean_std(t_vals)
        avg_tok = round(statistics.mean(tok_vals)) if tok_vals else 0

        row = f"| {label} | {d_m}\u00b1{d_s} | {o_m}\u00b1{o_s} | {t_m} | {avg_tok} | {len(runs)} |"
        ranked.append((label, o_m, row))

    # Sort by overall throughput descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    for _, _, row in ranked:
        lines.append(row)

    winner = ranked[0][0] if ranked else "?"
    lines.append("")
    lines.append(f"**Winner:** `{winner}` ({ranked[0][1]:.1f} tok/s)")

    if errors:
        lines.append("")
        lines.append(f"<details><summary>Errors ({len(errors)})</summary>")
        lines.append("")
        for e in errors[:10]:
            lines.append(f"- `{e.get('scenario', '?')}` [{e['server_config'].get('config_label', '?')}]: {e.get('error', '?')[:120]}")
        lines.append("</details>")

    lines.append("")
    return lines


def _section_dry(results: list[dict]) -> list[str]:
    """Generate Phase 3 DRY comparison."""
    p3 = [r for r in results if r.get("phase") == 3]
    ok = _ok_results(p3)

    if not ok:
        return ["## Phase 3: DRY Sampling", "", "*No data*", ""]

    lines = ["## Phase 3: DRY Sampling", ""]

    dry_off = [r for r in ok if not r["request_config"]["dry"]]
    dry_on = [r for r in ok if r["request_config"]["dry"]]

    off_tps = [r["metrics"]["overall_tps"] for r in dry_off]
    on_tps = [r["metrics"]["overall_tps"] for r in dry_on]

    off_m, off_s = _mean_std(off_tps)
    on_m, on_s = _mean_std(on_tps)

    lines.append("| DRY | Overall tok/s | Runs |")
    lines.append("|-----|--------------|------|")
    lines.append(f"| Off | {off_m}\u00b1{off_s} | {len(dry_off)} |")
    lines.append(f"| On  | {on_m}\u00b1{on_s} | {len(dry_on)} |")

    if off_m > 0:
        delta = _pct_change(off_m, on_m)
        lines.append("")
        lines.append(f"**Impact:** {delta} throughput with DRY enabled.")
        recommend = "on" if on_m >= off_m * 0.97 else "off"
        lines.append(f"**Recommendation:** DRY {recommend} (quality benefit vs throughput cost).")

    lines.append("")
    return lines


def _section_stress(results: list[dict]) -> list[str]:
    """Generate Phase 5 stress test summary."""
    p5 = [r for r in results if r.get("phase") == 5]
    ok = _ok_results(p5)

    if not ok:
        return ["## Phase 5: Stress Tests", "", "*No data*", ""]

    lines = ["## Phase 5: Stress Tests", ""]

    seq = [r for r in ok if not r.get("parallel_n")]
    par = [r for r in ok if r.get("parallel_n")]

    if seq:
        lines.append("### Sequential")
        lines.append("")
        lines.append("| Test | tok/s | TTFT (s) | Runs |")
        lines.append("|------|-------|----------|------|")
        by_name: dict[str, list[dict]] = {}
        for r in seq:
            by_name.setdefault(r["scenario"], []).append(r)
        for name, runs in sorted(by_name.items()):
            tps = [r["metrics"]["overall_tps"] for r in runs if r["metrics"].get("overall_tps")]
            ttft = [r["metrics"]["ttft_s"] for r in runs if r["metrics"].get("ttft_s")]
            if tps:
                lines.append(f"| {name} | {statistics.mean(tps):.1f} | {statistics.mean(ttft):.3f} | {len(runs)} |")
        lines.append("")

    if par:
        lines.append("### Parallel")
        lines.append("")
        lines.append("| Test | Slots | Agg tok/s | Per-req tok/s | Wall (s) | Runs |")
        lines.append("|------|-------|-----------|--------------|----------|------|")
        by_name = {}
        for r in par:
            by_name.setdefault(r["scenario"], []).append(r)
        for name, runs in sorted(by_name.items()):
            n = runs[0].get("parallel_n", "?")
            agg = [r["metrics"]["aggregate_tps"] for r in runs if r["metrics"].get("aggregate_tps")]
            per = [r["metrics"]["per_request_tps_mean"] for r in runs if r["metrics"].get("per_request_tps_mean")]
            wall = [r["metrics"]["wall_time_s"] for r in runs if r["metrics"].get("wall_time_s")]
            if agg:
                lines.append(
                    f"| {name} | {n} | {statistics.mean(agg):.1f} | "
                    f"{statistics.mean(per):.1f} | {statistics.mean(wall):.1f} | {len(runs)} |"
                )
        lines.append("")

    return lines


def _section_recommendation(results: list[dict]) -> list[str]:
    """Generate final recommendation section."""
    lines = ["## Recommendation", ""]

    # Collect winners
    p1_winner, p1_tps = _pick_phase_winner(results, 1, "config_label")
    p2_winner, p2_tps = _pick_phase_winner(results, 2, "config_label")

    # DRY
    p3_ok = _ok_results([r for r in results if r.get("phase") == 3])
    dry_off_tps = [r["metrics"]["overall_tps"] for r in p3_ok if not r["request_config"]["dry"]]
    dry_on_tps = [r["metrics"]["overall_tps"] for r in p3_ok if r["request_config"]["dry"]]
    if dry_off_tps and dry_on_tps:
        off_m, on_m = statistics.mean(dry_off_tps), statistics.mean(dry_on_tps)
        dry_rec = "on" if on_m >= off_m * 0.97 else "off"
    else:
        dry_rec = "default"

    lines.append("### Optimal Configuration")
    lines.append("")
    lines.append("```ini")
    lines.append("; models.ini — optimized settings from benchmark")
    if p1_winner:
        lines.append(f"; Speculative decoding: {p1_winner} ({p1_tps:.1f} tok/s)")
    if p2_winner:
        lines.append(f"; KV cache: {p2_winner} ({p2_tps:.1f} tok/s)")
    lines.append(f"; DRY sampling: {dry_rec}")
    lines.append("```")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(jsonl_path: Path, output_path: Path) -> None:
    """Generate full Markdown report from JSONL results."""
    results = load_results(jsonl_path)
    if not results:
        output_path.write_text("# Benchmark Report\n\nNo results found.\n")
        return

    ok = _ok_results(results)
    errors = [r for r in results if r.get("status") != "ok"]
    phases = sorted(set(r.get("phase", 0) for r in results))

    lines: list[str] = []

    # Header
    ts = results[0].get("timestamp", "")[:19] if results else ""
    lines.append("# llama.cpp Benchmark Report")
    lines.append("")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **First run:** {ts}")
    lines.append(f"- **Total runs:** {len(results)} ({len(ok)} ok, {len(errors)} errors)")
    lines.append(f"- **Phases:** {', '.join(str(p) for p in phases)}")
    lines.append("")

    # Phase sections
    if 1 in phases:
        lines.extend(_section_phase(results, 1, "Speculative Decoding", "config_label"))
    if 2 in phases:
        lines.extend(_section_phase(results, 2, "KV Cache Dtype", "config_label"))
    if 3 in phases:
        lines.extend(_section_dry(results))
    if 4 in phases:
        lines.extend(_section_phase(results, 4, "Cross-Validation", "config_label"))
    if 5 in phases:
        lines.extend(_section_stress(results))

    # Recommendation
    lines.extend(_section_recommendation(results))

    output_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_report.py <path_to_runs.jsonl>")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        sys.exit(1)

    report_path = jsonl_path.with_name(jsonl_path.stem.replace("_runs", "_report") + ".md")
    generate_report(jsonl_path, report_path)
    print(f"Report written to {report_path}")
