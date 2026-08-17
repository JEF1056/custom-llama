#!/usr/bin/env python3
"""
Generate a publication-grade comparison chart from benchmark_results.json.
Visualizes:
  1. Prefill Processing Speed (tok/s) across 2k, 50k, 125k
  2. Autoregressive Decode Speed (tok/s)
  3. Prefix Cache TTFT Reduction Speedup (x)
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_chart(json_path: str, output_img: str):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Color palette & styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    host_labels = {
        "http://ml-1-wsl:8080/v1": "ml-1-wsl (RTX 3090 / CUDA)",
        "http://ml-2:8080/v1": "ml-2 (Apple M5 Pro)",
        "http://ml-3:8080/v1": "ml-3 (Apple M4 Pro)",
    }
    colors = {
        "http://ml-1-wsl:8080/v1": "#2b5c8f",
        "http://ml-2:8080/v1": "#2e7d32",
        "http://ml-3:8080/v1": "#d84315",
    }

    tiers = ["2k", "50k", "125k"]
    tier_labels = ["2K Tokens", "50K Tokens", "125K Tokens"]
    x = np.arange(len(tiers))
    width = 0.25

    # 1. Prefill Speed (tok/s)
    ax1 = axes[0]
    for i, (host_key, host_name) in enumerate(host_labels.items()):
        h_data = data.get(host_key, {}).get("tiers", {})
        prefills = []
        for t in tiers:
            val = h_data.get(t, {}).get("cold", {}).get("prefill_tok_per_sec", 0)
            prefills.append(val)
        bars = ax1.bar(x + (i - 1) * width, prefills, width, label=host_name, color=colors[host_key], edgecolor='black', linewidth=0.5)
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax1.annotate(f'{int(yval)}',
                             xy=(bar.get_x() + bar.get_width() / 2, yval),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_title("Prompt Processing / Prefill Speed (tok/s)\n(Higher is Better)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_labels, fontsize=9, fontweight='bold')
    ax1.set_ylabel("Tokens / Second", fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=8)

    # 2. Decode Speed (tok/s)
    ax2 = axes[1]
    for i, (host_key, host_name) in enumerate(host_labels.items()):
        h_data = data.get(host_key, {}).get("tiers", {})
        decodes = []
        for t in tiers:
            val = h_data.get(t, {}).get("cold", {}).get("decode_tok_per_sec", 0)
            decodes.append(val)
        bars = ax2.bar(x + (i - 1) * width, decodes, width, label=host_name, color=colors[host_key], edgecolor='black', linewidth=0.5)
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax2.annotate(f'{yval:.1f}',
                             xy=(bar.get_x() + bar.get_width() / 2, yval),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_title("Decode Generation Speed (tok/s)\n(Higher is Better)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_labels, fontsize=9, fontweight='bold')
    ax2.set_ylabel("Tokens / Second", fontsize=10, fontweight='bold')

    # 3. Cache Hit Speedup (x TTFT Reduction)
    ax3 = axes[2]
    for i, (host_key, host_name) in enumerate(host_labels.items()):
        h_data = data.get(host_key, {}).get("tiers", {})
        speedups = []
        for t in tiers:
            c = h_data.get(t, {}).get("cold", {}).get("ttft_sec", 1.0)
            w = h_data.get(t, {}).get("warm", {}).get("ttft_sec", 1.0) if h_data.get(t, {}).get("warm") else 1.0
            sp = round(c / max(0.001, w), 1) if w > 0 else 1.0
            speedups.append(sp)
        bars = ax3.bar(x + (i - 1) * width, speedups, width, label=host_name, color=colors[host_key], edgecolor='black', linewidth=0.5)
        for bar in bars:
            yval = bar.get_height()
            if yval > 1:
                ax3.annotate(f'{yval:.0f}x',
                             xy=(bar.get_x() + bar.get_width() / 2, yval),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax3.set_title("Prefix Cache TTFT Speedup (x Factor)\n(Higher is Better)", fontsize=11, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(tier_labels, fontsize=9, fontweight='bold')
    ax3.set_ylabel("Speedup Multiplier", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    print(f"Comparison chart successfully generated: {output_img}")

if __name__ == "__main__":
    generate_comparison_chart(
        "/Users/jfan/Documents/GitHub/custom-llama/scripts/benchmark_results.json",
        "/Users/jfan/Documents/GitHub/custom-llama/scripts/nodes_comparison_chart.png"
    )
