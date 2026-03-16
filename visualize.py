#!/usr/bin/env python3
"""
llm-cpu-bench Visualizer — 캐시 벤치마크 결과 비교 차트 생성
=============================================================
사용법:
  python visualize.py 9800x3d.json 9700x.json
  python visualize.py *.json --output-dir ./charts
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required.")
    print("  pip install matplotlib numpy")
    sys.exit(1)

PALETTE = ["#7F77DD", "#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#639922", "#BA7517", "#E24B4A"]
GRID = "#E5E5E5"
TEXT2 = "#888780"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.grid": True,
        "grid.color": GRID, "grid.alpha": 0.5, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "medium",
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load(filepath):
    with open(filepath) as f:
        return json.load(f)


def get_label(data, filepath):
    return data.get("meta", {}).get("label", Path(filepath).stem)


# ─────────────────────────────────────────────
# Chart 1: Cache Latency Profile (핵심 차트)
# ─────────────────────────────────────────────

def chart_cache_latency(datasets, labels, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (data, label) in enumerate(zip(datasets, labels)):
        results = data.get("results", {}).get("cache_latency", [])
        if not results:
            continue
        sizes = [r["size_mb"] for r in results]
        latencies = [r["ns_per_access"] for r in results]
        ax.plot(sizes, latencies, "o-", color=PALETTE[i % len(PALETTE)],
                label=label, linewidth=2, markersize=5, zorder=3)

    # L3 경계 표시
    ax.axvline(x=32, color="#E24B4A", linewidth=1.5, linestyle="--", alpha=0.7, label="32 MB (Non-X3D L3)")
    ax.axvline(x=96, color="#7F77DD", linewidth=1.5, linestyle="--", alpha=0.7, label="96 MB (X3D L3)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Buffer size (MB)")
    ax.set_ylabel("Latency (ns/access)")
    ax.set_title("Cache Latency Profile — Where X3D Wins")
    ax.legend(fontsize=9, loc="upper left")

    # X축 라벨
    tick_positions = [0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 192]
    tick_labels = []
    for t in tick_positions:
        if t >= 1:
            tick_labels.append(f"{t:.0f}")
        else:
            tick_labels.append(f"{t}")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    fp = output_dir / "01_cache_latency_profile.png"
    fig.savefig(fp)
    plt.close(fig)
    print(f"  * {fp}")


# ─────────────────────────────────────────────
# Chart 2: Tokenizer Lookup Performance
# ─────────────────────────────────────────────

def chart_tokenizer(datasets, labels, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    all_vocab_names = []
    for data in datasets:
        for r in data.get("results", {}).get("tokenizer_lookup", []):
            if r["vocab_name"] not in all_vocab_names:
                all_vocab_names.append(r["vocab_name"])

    if not all_vocab_names:
        return

    x = np.arange(len(all_vocab_names))
    n = len(datasets)
    bar_w = 0.7 / n

    for i, (data, label) in enumerate(zip(datasets, labels)):
        results = {r["vocab_name"]: r for r in data.get("results", {}).get("tokenizer_lookup", [])}
        vals = [results.get(name, {}).get("lookups_per_sec", 0) for name in all_vocab_names]
        offset = (i - (n - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.85, label=label,
                      color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:,.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=TEXT2)

    ax.set_ylabel("Lookups / second")
    ax.set_title("Tokenizer Vocab Lookup — Tokenize/Detokenize Speed")
    ax.set_xticks(x)
    ax.set_xticklabels(all_vocab_names, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fp = output_dir / "02_tokenizer_lookup.png"
    fig.savefig(fp)
    plt.close(fig)
    print(f"  * {fp}")


# ─────────────────────────────────────────────
# Chart 3: RAG Vector Search
# ─────────────────────────────────────────────

def chart_rag_search(datasets, labels, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    all_configs = []
    for data in datasets:
        for r in data.get("results", {}).get("rag_vector_search", []):
            name = f"{r['n_vectors']//1000}K x {r['dimension']}d\n({r['db_size_mb']:.0f} MB)"
            if name not in all_configs:
                all_configs.append(name)

    if not all_configs:
        return

    x = np.arange(len(all_configs))
    n = len(datasets)
    bar_w = 0.7 / n

    for i, (data, label) in enumerate(zip(datasets, labels)):
        rag_results = data.get("results", {}).get("rag_vector_search", [])
        vals = []
        for config_name in all_configs:
            found = False
            for r in rag_results:
                name = f"{r['n_vectors']//1000}K x {r['dimension']}d\n({r['db_size_mb']:.0f} MB)"
                if name == config_name:
                    vals.append(r["queries_per_sec"])
                    found = True
                    break
            if not found:
                vals.append(0)

        offset = (i - (n - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.85, label=label,
                      color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=TEXT2)

    ax.set_ylabel("Queries / second")
    ax.set_title("RAG Vector Search — Embedding Similarity Retrieval")
    ax.set_xticks(x)
    ax.set_xticklabels(all_configs, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fp = output_dir / "03_rag_vector_search.png"
    fig.savefig(fp)
    plt.close(fig)
    print(f"  * {fp}")


# ─────────────────────────────────────────────
# Chart 4: Relative Performance Summary
# ─────────────────────────────────────────────

def chart_summary(datasets, labels, output_dir):
    if len(datasets) < 2:
        return

    base = datasets[0]
    base_label = labels[0]
    fig, ax = plt.subplots(figsize=(10, max(4, len(datasets) * 2)))

    categories = []
    all_diffs = {label: [] for label in labels[1:]}

    # 캐시 레이턴시: 32MB와 96MB 지점 비교 (낮을수록 좋음 → 반전)
    for size_target, name in [(32, "Cache latency @ 32MB"), (96, "Cache latency @ 96MB")]:
        base_val = None
        for r in base.get("results", {}).get("cache_latency", []):
            if abs(r["size_mb"] - size_target) < 1:
                base_val = r["ns_per_access"]
                break
        if base_val is None:
            continue
        categories.append(name)
        for data, label in zip(datasets[1:], labels[1:]):
            comp_val = None
            for r in data.get("results", {}).get("cache_latency", []):
                if abs(r["size_mb"] - size_target) < 1:
                    comp_val = r["ns_per_access"]
                    break
            if comp_val and base_val:
                # 레이턴시는 낮을수록 좋음 → 반전 (base가 빠르면 양수)
                diff = (comp_val - base_val) / comp_val * 100
                all_diffs[label].append(diff)
            else:
                all_diffs[label].append(0)

    # 토크나이저: lookups/sec (높을수록 좋음)
    for i, r_base in enumerate(base.get("results", {}).get("tokenizer_lookup", [])):
        categories.append(f"Tokenizer: {r_base['vocab_name']}")
        for data, label in zip(datasets[1:], labels[1:]):
            r_comp_list = data.get("results", {}).get("tokenizer_lookup", [])
            if i < len(r_comp_list):
                comp_val = r_comp_list[i]["lookups_per_sec"]
                base_val = r_base["lookups_per_sec"]
                if comp_val > 0:
                    diff = (base_val - comp_val) / comp_val * 100
                    all_diffs[label].append(diff)
                else:
                    all_diffs[label].append(0)
            else:
                all_diffs[label].append(0)

    # RAG 검색: queries/sec (높을수록 좋음)
    for i, r_base in enumerate(base.get("results", {}).get("rag_vector_search", [])):
        categories.append(f"RAG: {r_base['n_vectors']//1000}K x {r_base['dimension']}d")
        for data, label in zip(datasets[1:], labels[1:]):
            r_comp_list = data.get("results", {}).get("rag_vector_search", [])
            if i < len(r_comp_list):
                comp_val = r_comp_list[i]["queries_per_sec"]
                base_val = r_base["queries_per_sec"]
                if comp_val > 0:
                    diff = (base_val - comp_val) / comp_val * 100
                    all_diffs[label].append(diff)
                else:
                    all_diffs[label].append(0)
            else:
                all_diffs[label].append(0)

    if not categories:
        return

    y = np.arange(len(categories))
    bar_h = 0.7 / len(all_diffs)

    for i, (label, diffs) in enumerate(all_diffs.items()):
        offset = (i - (len(all_diffs) - 1) / 2) * bar_h
        colors = ["#1D9E75" if d >= 0 else "#E24B4A" for d in diffs]
        bars = ax.barh(y + offset, diffs, bar_h * 0.85, color=colors,
                       edgecolor="white", linewidth=0.5, zorder=3, label=f"{base_label} vs {label}")
        for j, (bar, diff) in enumerate(zip(bars, diffs)):
            w = bar.get_width()
            side = 3 if w >= 0 else -3
            ha = "left" if w >= 0 else "right"
            ax.annotate(f"{diff:+.1f}%", xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(side, 0), textcoords="offset points",
                        ha=ha, va="center", fontsize=8, fontweight="medium")

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel(f"Performance advantage of {base_label} (%)")
    ax.set_title(f"Overall: {base_label} vs Others")
    ax.axvline(x=0, color="#2C2C2A", linewidth=0.8, zorder=2)
    ax.invert_yaxis()
    if len(all_diffs) > 1:
        ax.legend(fontsize=8)

    fp = output_dir / "04_summary.png"
    fig.savefig(fp)
    plt.close(fig)
    print(f"  * {fp}")


# ─────────────────────────────────────────────
# Chart 0: System Info Table
# ─────────────────────────────────────────────

def chart_system_info(datasets, labels, output_dir):
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(labels)), 4))
    ax.axis("off")

    def gv(d, *keys):
        obj = d
        for k in keys:
            obj = obj.get(k, "-") if isinstance(obj, dict) else "-"
        return str(obj) if obj else "-"

    fields = [
        ("CPU", lambda d: gv(d, "system", "cpu", "model")),
        ("L3 Cache", lambda d: gv(d, "system", "cpu", "l3_cache")),
        ("Cores", lambda d: f"{gv(d, 'system', 'cpu', 'cores_physical')}P / {gv(d, 'system', 'cpu', 'cores_logical')}T"),
        ("Memory", lambda d: f"{gv(d, 'system', 'memory', 'total_gb')} GB @ {gv(d, 'system', 'memory', 'speed')}"),
        ("OS", lambda d: f"{gv(d, 'system', 'os', 'system')} {gv(d, 'system', 'os', 'release')}"),
    ]

    rows = [[fn, *[f(d) for d in datasets]] for fn, f in fields]
    col_labels = [""] + labels
    col_colors = [GRID] + [PALETTE[i % len(PALETTE)] + "30" for i in range(len(labels))]
    cell_colors = [["#F5F5F5" if i % 2 == 0 else "white"] * len(col_labels) for i in range(len(rows))]

    table = ax.table(cellText=rows, colLabels=col_labels, cellColours=cell_colors,
                     colColours=col_colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("System Configuration", fontweight="medium", pad=20)

    fp = output_dir / "00_system_info.png"
    fig.savefig(fp)
    plt.close(fig)
    print(f"  * {fp}")


# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="llm-cpu-bench: Compare cache benchmark results")
    parser.add_argument("files", nargs="+", help="Result JSON files")
    parser.add_argument("--output-dir", "-o", default="./charts", help="Chart output directory")
    args = parser.parse_args()

    for f in args.files:
        if not Path(f).exists():
            print(f"ERROR: {f} not found")
            sys.exit(1)

    datasets = [load(f) for f in args.files]
    labels = [get_label(d, f) for d, f in zip(datasets, args.files)]

    print(f"Comparing {len(datasets)} results:")
    for label, f in zip(labels, args.files):
        print(f"  [{label}] {f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"\nGenerating charts...")
    chart_system_info(datasets, labels, output_dir)
    chart_cache_latency(datasets, labels, output_dir)
    chart_tokenizer(datasets, labels, output_dir)
    chart_rag_search(datasets, labels, output_dir)
    chart_summary(datasets, labels, output_dir)

    print(f"\nDone! Charts in: {output_dir}/")
    print(f"  00_system_info.png           — System specs")
    print(f"  01_cache_latency_profile.png — Cache latency (the key chart)")
    print(f"  02_tokenizer_lookup.png      — Tokenize/Detokenize speed")
    print(f"  03_rag_vector_search.png     — RAG retrieval speed")
    print(f"  04_summary.png               — Overall % comparison")


if __name__ == "__main__":
    main()
