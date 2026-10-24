"""
visualization/pareto.py
=======================
Pareto frontier analysis and visualization.

Computes which model configurations are non-dominated
(Pareto-optimal) across accuracy and resource cost dimensions,
then generates publication-quality plots.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pareto computation
# ---------------------------------------------------------------------------

def find_pareto_frontier(accuracy: list, cost: list) -> list[int]:
    """
    Find non-dominated points in (maximize accuracy, minimize cost) space.

    A point i is dominated if there exists j such that:
        accuracy[j] >= accuracy[i]  AND  cost[j] <= cost[i]
        with at least one strict inequality.

    Returns:
        List of indices of non-dominated (Pareto-optimal) points.
    """
    n = len(accuracy)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Does j dominate i?
            if (accuracy[j] >= accuracy[i] and
                    cost[j] <= cost[i] and
                    (accuracy[j] > accuracy[i] or cost[j] < cost[i])):
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Colour scheme (consistent across all plots)
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "baseline":     "#2c2c2c",     # near-black
    "pruning":      "#3a86ff",     # blue
    "quantization": "#ff006e",     # red/pink
    "low_rank":     "#38b000",     # green
}

METHOD_MARKERS = {
    "baseline":     "D",
    "pruning":      "o",
    "quantization": "s",
    "low_rank":     "^",
}


# ---------------------------------------------------------------------------
# Single 2D Pareto plot
# ---------------------------------------------------------------------------

def plot_pareto_2d(
    configs: list[dict],
    accuracy: list[float],
    cost: list[float],
    cost_label: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    log_x: bool = False,
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> plt.Figure:
    """
    Scatter plot with Pareto frontier overlaid.

    Args:
        configs:     list of dicts, each must have 'method' key
        accuracy:    list of accuracy values (0–1 or 0–100)
        cost:        list of cost values (to minimize)
        cost_label:  x-axis label
        title:       plot title (auto-generated if None)
        output_path: save path (PNG), skips save if None
        log_x:       use log scale for x-axis

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scale accuracy to %
    acc_pct = [a * 100 if a <= 1.0 else a for a in accuracy]

    # Group by method for legend
    plotted_methods = set()
    for i, cfg in enumerate(configs):
        method = cfg.get("method", "baseline")
        color  = METHOD_COLORS.get(method, "#888888")
        marker = METHOD_MARKERS.get(method, "o")
        label  = method.replace("_", " ").title() if method not in plotted_methods else "_nolegend_"
        ax.scatter(cost[i], acc_pct[i],
                   c=color, marker=marker, s=80, zorder=3,
                   label=label, edgecolors="white", linewidths=0.5)
        plotted_methods.add(method)

    # Pareto frontier
    pareto_idx = find_pareto_frontier(acc_pct, cost)
    if len(pareto_idx) > 1:
        pf_sorted = sorted(pareto_idx, key=lambda i: cost[i])
        pf_x = [cost[i]    for i in pf_sorted]
        pf_y = [acc_pct[i] for i in pf_sorted]
        ax.plot(pf_x, pf_y, "k--", linewidth=1.5, zorder=2, label="Pareto frontier")

        # Shade dominated region (below-right of frontier)
        ax.fill_betweenx(
            y=[min(acc_pct) - 1, pf_y[0]],
            x1=pf_x[0],  x2=max(cost) * 1.05,
            alpha=0.04, color="gray"
        )

    # Formatting
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(cost_label, fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(title or f"Accuracy vs {cost_label}", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[pareto] Saved -> {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Full dashboard (2x3 subplot)
# ---------------------------------------------------------------------------

def plot_pareto_dashboard(
    configs: list[dict],
    accuracy: list[float],
    model_size_mb: list[float],
    latency_ms: list[float],
    ram_mb: list[float],
    flops_m: list[float],
    params: list[int],
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    2x3 dashboard: Accuracy vs (Size, Latency, RAM, FLOPs, Params).
    """
    acc_pct = [a * 100 if a <= 1.0 else a for a in accuracy]

    panels = [
        (model_size_mb, "Model Size (MB)",        False),
        (latency_ms,    "Inference Latency (ms)", False),
        (ram_mb,        "Peak RAM (MB)",           False),
        (flops_m,       "FLOPs (M)",               True),
        (params,        "Parameters",              True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_idx, (cost_vals, xlabel, log_x) in enumerate(panels):
        ax = axes[ax_idx]

        plotted = set()
        for i, cfg in enumerate(configs):
            method = cfg.get("method", "baseline")
            color  = METHOD_COLORS.get(method, "#888888")
            marker = METHOD_MARKERS.get(method, "o")
            lbl    = method.replace("_", " ").title() if method not in plotted else "_nolegend_"
            ax.scatter(cost_vals[i], acc_pct[i],
                       c=color, marker=marker, s=60, zorder=3,
                       label=lbl, edgecolors="white", linewidths=0.4)
            plotted.add(method)

        # Pareto frontier
        pf = find_pareto_frontier(acc_pct, cost_vals)
        if len(pf) > 1:
            pf_sorted = sorted(pf, key=lambda i: cost_vals[i])
            ax.plot([cost_vals[i] for i in pf_sorted],
                    [acc_pct[i]  for i in pf_sorted],
                    "k--", linewidth=1.2, zorder=2)

        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    # FLOPs vs Latency (RQ2 plot) in the 6th panel
    ax = axes[5]
    for i, cfg in enumerate(configs):
        method = cfg.get("method", "baseline")
        color  = METHOD_COLORS.get(method, "#888888")
        marker = METHOD_MARKERS.get(method, "o")
        ax.scatter(flops_m[i], latency_ms[i],
                   c=color, marker=marker, s=60, zorder=3,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel("FLOPs (M)", fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title("FLOPs vs Actual Latency (RQ2)", fontsize=10, fontstyle="italic")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Accuracy–Resource Trade-off Dashboard\n"
        "(Dashed line = Pareto frontier)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[pareto] Dashboard saved -> {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Accuracy degradation vs compression ratio
# ---------------------------------------------------------------------------

def plot_accuracy_vs_compression_ratio(
    methods_data: dict,
    baseline_accuracy: float,
    baseline_size_mb: float,
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot accuracy drop (%) vs compression ratio for each method.

    Args:
        methods_data: dict mapping method_name -> list of (size_mb, accuracy) tuples
        baseline_accuracy: float, test accuracy of uncompressed model
        baseline_size_mb:  float, uncompressed model size in MB
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    base_acc_pct = baseline_accuracy * 100 if baseline_accuracy <= 1 else baseline_accuracy

    for method, data in methods_data.items():
        sizes_mb = [d[0] for d in data]
        accs     = [d[1] * 100 if d[1] <= 1 else d[1] for d in data]

        comp_ratios = [baseline_size_mb / s for s in sizes_mb]
        acc_drops   = [base_acc_pct - a for a in accs]

        color  = METHOD_COLORS.get(method, "#888888")
        marker = METHOD_MARKERS.get(method, "o")
        ax.plot(comp_ratios, acc_drops,
                color=color, marker=marker, linewidth=2,
                label=method.replace("_", " ").title())

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Compression Ratio (baseline_size / compressed_size)", fontsize=12)
    ax.set_ylabel("Accuracy Drop (%)", fontsize=12)
    ax.set_title("Accuracy Degradation vs Compression Ratio", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[pareto] Saved -> {output_path}")

    return fig


def main(args_list=None):
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate Pareto frontier plots from benchmark results")
    parser.add_argument("--csv", default=None, help="Path to benchmark_all.csv or results directory")
    parser.add_argument("--output_dir", default="benchmarks/figures", help="Directory to save figures")
    args = parser.parse_args(args_list)

    results_dir = Path("benchmarks/results")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = None
    if args.csv and Path(args.csv).is_file():
        df = pd.read_csv(args.csv)
    elif (results_dir / "benchmark_all.csv").is_file():
        df = pd.read_csv(results_dir / "benchmark_all.csv")
    else:
        # Aggregate whatever individual CSVs exist
        dfs = []
        for name in ["baseline.csv", "pruning_sweep.csv", "quantization_sweep.csv"]:
            p = results_dir / name
            if p.is_file():
                dfs.append(pd.read_csv(p))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)

    if df is None or len(df) == 0:
        print("[pareto] No benchmark results found to plot.")
        return

    print(f"[pareto] Loaded {len(df)} experiment rows.")

    # Deduplicate or average if multiple seeds exist per configuration
    configs = df.to_dict("records")
    accuracy = df["test_accuracy"].tolist()
    size_mb  = df["model_size_param_mb"].tolist()
    latency  = df["latency_mean_ms"].tolist()
    ram_mb   = df["peak_ram_mb"].tolist()
    flops_m  = df["MACs_M"].tolist()
    params   = df["params_trainable"].tolist()

    # 2. Individual 2D Pareto plots
    plot_pareto_2d(
        configs, accuracy, size_mb,
        cost_label="Model Size (MB)",
        title="Pareto: Accuracy vs Model Size",
        output_path=str(output_dir / "pareto_accuracy_vs_size.png"),
    )
    plot_pareto_2d(
        configs, accuracy, latency,
        cost_label="Inference Latency (ms)",
        title="Pareto: Accuracy vs Latency",
        output_path=str(output_dir / "pareto_accuracy_vs_latency.png"),
    )
    plot_pareto_2d(
        configs, accuracy, ram_mb,
        cost_label="Peak RAM (MB)",
        title="Pareto: Accuracy vs Peak RAM",
        output_path=str(output_dir / "pareto_accuracy_vs_ram.png"),
    )

    # 3. 6-panel Dashboard
    plot_pareto_dashboard(
        configs, accuracy, size_mb, latency, ram_mb, flops_m, params,
        output_path=str(output_dir / "pareto_dashboard.png"),
    )

    # 4. Accuracy degradation vs compression ratio
    baseline_rows = df[df["method"] == "baseline"]
    if len(baseline_rows) > 0:
        base_acc = float(baseline_rows["test_accuracy"].iloc[0])
        base_size = float(baseline_rows["model_size_param_mb"].iloc[0])

        methods_data = {}
        for method, group in df.groupby("method"):
            methods_data[method] = list(zip(group["model_size_param_mb"], group["test_accuracy"]))

        plot_accuracy_vs_compression_ratio(
            methods_data,
            baseline_accuracy=base_acc,
            baseline_size_mb=base_size,
            output_path=str(output_dir / "accuracy_vs_compression_ratio.png"),
        )

    print(f"[pareto] All figures saved to {output_dir}")


if __name__ == "__main__":
    main()

