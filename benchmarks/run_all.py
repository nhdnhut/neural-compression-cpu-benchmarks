"""
benchmarks/run_all.py
=====================
Master script: runs all experiments in sequence and merges results
into a single benchmark CSV + summary JSON.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --seeds 42 --skip_pruning

Order:
    1. Baseline (E00)
    2. Pruning sweep (E01–E04)
    3. Quantization sweep (E05)

The script assumes baseline models have been saved before running sweeps.
If baseline models don't exist, it trains them first.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from experiments.baseline import run_baseline
from experiments.pruning_sweep import run_pruning_experiment
from experiments.quantization_sweep import run_quantization_experiment
from experiments.low_rank_sweep import run_low_rank_experiment
from profiling.reporter import save_report_csv


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark suite")
    parser.add_argument("--baseline_config", default="configs/baseline.yaml")
    parser.add_argument("--pruning_config",  default="configs/pruning.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip_baseline",    action="store_true")
    parser.add_argument("--skip_pruning",     action="store_true")
    parser.add_argument("--skip_quantization",action="store_true")
    parser.add_argument("--skip_low_rank",    action="store_true")
    parser.add_argument("--finetune_epochs",  type=int, default=None)
    args = parser.parse_args()

    with open(args.baseline_config) as f:
        baseline_config = yaml.safe_load(f)
    with open(args.pruning_config) as f:
        pruning_config = yaml.safe_load(f)

    if args.finetune_epochs is not None:
        pruning_config["pruning"]["finetune_epochs"] = args.finetune_epochs

    seeds       = args.seeds or baseline_config["seeds"]
    results_dir = Path(baseline_config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir  = Path(baseline_config["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # -- 1. Baseline ---------------------------------------------------------
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("  PHASE 1: Baseline (E00)")
        print("="*60)
        for seed in seeds:
            r = run_baseline(baseline_config, seed=seed, verbose=True)
            all_results.append(r)
    else:
        baseline_csv = results_dir / "baseline.csv"
        if baseline_csv.is_file():
            print(f"[run_all] Loading existing baseline results from {baseline_csv}")
            all_results.extend(pd.read_csv(baseline_csv).to_dict("records"))

    # -- 2. Pruning sweep ----------------------------------------------------
    if not args.skip_pruning:
        print("\n" + "="*60)
        print("  PHASE 2: Pruning Sweep (E01–E04)")
        print("="*60)
        ratios = pruning_config["pruning"]["ratios"]
        # Skip 0.0 — same as baseline
        for seed in seeds:
            for ratio in [r for r in ratios if r > 0]:
                r = run_pruning_experiment(
                    pruning_config, baseline_config,
                    seed=seed, pruning_ratio=ratio, verbose=True,
                )
                all_results.append(r)
    else:
        pruning_csv = results_dir / "pruning_sweep.csv"
        if pruning_csv.is_file():
            print(f"[run_all] Loading existing pruning results from {pruning_csv}")
            all_results.extend(pd.read_csv(pruning_csv).to_dict("records"))

    # -- 3. Quantization -----------------------------------------------------
    if not args.skip_quantization:
        print("\n" + "="*60)
        print("  PHASE 3: Quantization (E05)")
        print("="*60)
        for seed in seeds:
            r = run_quantization_experiment(baseline_config, seed=seed, method="dynamic_int8")
            all_results.append(r)
    else:
        quant_csv = results_dir / "quantization_sweep.csv"
        if quant_csv.is_file():
            print(f"[run_all] Loading existing quantization results from {quant_csv}")
            all_results.extend(pd.read_csv(quant_csv).to_dict("records"))

    # -- 4. Low-rank SVD sweep -----------------------------------------------
    if not args.skip_low_rank:
        print("\n" + "="*60)
        print("  PHASE 4: Low-Rank SVD Sweep (E06–E09)")
        print("="*60)
        lr_ratios = [0.25, 0.50, 0.75]   # 1.0 = same as baseline, skip
        for seed in seeds:
            for ratio in lr_ratios:
                r = run_low_rank_experiment(
                    baseline_config, seed=seed,
                    rank_ratio=ratio, verbose=True,
                )
                all_results.append(r)
    else:
        lr_csv = results_dir / "low_rank_sweep.csv"
        if lr_csv.is_file():
            print(f"[run_all] Loading existing low-rank results from {lr_csv}")
            all_results.extend(pd.read_csv(lr_csv).to_dict("records"))

    # -- Merge + Save --------------------------------------------------------
    if all_results:
        master_csv = str(results_dir / "benchmark_all.csv")
        save_report_csv(all_results, master_csv)

        # Aggregate summary: mean +/- std per (method, pruning_ratio, quantization)
        df = pd.DataFrame(all_results)
        numeric_cols = ["test_accuracy", "test_macro_f1", "latency_mean_ms",
                        "model_size_param_mb", "MACs_M", "global_sparsity",
                        "params_trainable"]
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        summary = (
            df.groupby(["method", "pruning_ratio", "quantization"])[numeric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary_path = str(results_dir / "benchmark_summary.csv")
        summary.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print("  BENCHMARK COMPLETE")
        print(f"  Results:  {master_csv}")
        print(f"  Summary:  {summary_path}")
        print(f"  Total experiments: {len(all_results)}")
        print(f"{'='*60}\n")

        # Automatically update Pareto visualizations
        try:
            from visualization.pareto import main as generate_pareto
            generate_pareto([])
        except Exception as e:
            print(f"[run_all] Warning: Could not generate Pareto figures: {e}")
    else:
        print("[run_all] No experiments were run (all phases skipped).")


if __name__ == "__main__":
    main()

