"""
experiments/low_rank_sweep.py
==============================
Run SVD low-rank factorization experiments (E06–E09):
  Load baseline model, apply truncated SVD at multiple rank ratios,
  evaluate, profile, save results.

Method: Replace W (m x n) with U_r @ diag(S_r) @ V_r^T (rank-r reconstruction).
  This preserves layer architecture (same shape) but introduces information loss
  proportional to the discarded singular values.

  NOTE: This variant keeps the same layer architecture (reconstruction, not
  factorisation). Real parameter reduction requires replacing one Linear(m,n)
  with two: Linear(m,r) -> Linear(r,n). That extension is left for future work
  and documented in the paper as a limitation.

Usage:
    python experiments/low_rank_sweep.py
    python experiments/low_rank_sweep.py --seeds 42 --ratios 0.25 0.5 0.75 1.0
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn.architectures import SmallCNN
from data.loaders import load_fashion_mnist
from optim.trainer import evaluate, set_seed
from compress.low_rank import apply_svd_approximation, svd_energy_retained
from profiling.reporter import profile_model, save_report_csv, print_report


def run_low_rank_experiment(
    baseline_config: dict,
    seed: int,
    rank_ratio: float,
    verbose: bool = True,
) -> dict:
    """
    Load baseline model, apply SVD approximation, evaluate, profile.

    Args:
        baseline_config: config dict from baseline.yaml
        seed:            random seed
        rank_ratio:      fraction of singular values to keep (1.0 = no change)
        verbose:         print progress

    Returns:
        results dict (one row of the benchmark table)
    """
    exp_label = f"lowrank_r{int(rank_ratio * 100):03d}_seed{seed}"
    print(f"\n{'-'*60}")
    print(f"  {exp_label}")
    print(f"{'-'*60}")

    set_seed(seed)

    # ---- Data ---------------------------------------------------------------
    ds_cfg = baseline_config["dataset"]
    data   = load_fashion_mnist(
        data_dir     = ds_cfg["data_dir"],
        batch_size   = ds_cfg["batch_size"],
        val_fraction = ds_cfg["val_fraction"],
        seed         = seed,
    )

    # ---- Load baseline model -----------------------------------------------
    models_dir    = Path(baseline_config["output"]["models_dir"])
    baseline_path = str(models_dir / f"baseline_seed{seed}.pth")

    model = SmallCNN(
        num_classes = baseline_config["model"]["num_classes"],
        dropout_p   = baseline_config["model"]["dropout_p"],
    )
    state = torch.load(baseline_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded baseline from {baseline_path}")

    # ---- Apply SVD approximation -------------------------------------------
    if rank_ratio < 1.0:
        approx_model, svd_stats = apply_svd_approximation(
            model, rank_ratio=rank_ratio, copy_model=True
        )
        print(f"  Target rank ratio: {rank_ratio*100:.1f}%")
        for layer_name, s in svd_stats.items():
            print(f"    {layer_name}: rank {s['rank_r']}/{s['rank_full']}  "
                  f"error={s['approx_error']:.4f}  "
                  f"energy_retained={1 - s['approx_error']:.4f}")
    else:
        # rank_ratio=1.0: no compression (reference point, same as baseline)
        import copy
        approx_model = copy.deepcopy(model)
        svd_stats = {}
        print("  rank_ratio=1.0: no approximation (reference)")

    # ---- Compute mean approximation error and energy retention --------------
    if svd_stats:
        mean_approx_error   = float(np.mean([s["approx_error"]   for s in svd_stats.values()]))
        mean_energy_retained = float(np.mean(
            [1.0 - s["approx_error"] for s in svd_stats.values()]
        ))
        mean_param_ratio    = float(np.mean([s["param_ratio"]    for s in svd_stats.values()]))
    else:
        mean_approx_error    = 0.0
        mean_energy_retained = 1.0
        mean_param_ratio     = 1.0

    # ---- Evaluate -----------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    test_eval = evaluate(approx_model, data["test_loader"], criterion)
    print(f"  Test accuracy: {test_eval['accuracy']*100:.2f}%  |  "
          f"F1: {test_eval['macro_f1']*100:.2f}%")

    # ---- Save approximated model --------------------------------------------
    save_path = str(models_dir / f"lowrank_r{int(rank_ratio*100):03d}_seed{seed}.pth")
    torch.save(approx_model.state_dict(), save_path)

    # ---- Profile ------------------------------------------------------------
    p_cfg = baseline_config["profiling"]
    profile = profile_model(
        model                  = approx_model.eval(),
        input_shape_single     = tuple(p_cfg["input_shape"]),
        batch_size_throughput  = p_cfg["batch_size_throughput"],
        latency_warmup         = p_cfg["latency_warmup"],
        latency_runs           = p_cfg["latency_runs"],
        saved_path             = save_path,
        label                  = exp_label,
    )
    print_report(profile)

    result = {
        "exp_id":               f"E_lr{int(rank_ratio * 100):03d}",
        "method":               "low_rank",
        "pruning_ratio":        0.0,
        "quantization":         "fp32",
        "rank_ratio":           rank_ratio,
        "seed":                 seed,
        "dataset":              "FashionMNIST",
        "test_accuracy":        test_eval["accuracy"],
        "test_macro_f1":        test_eval["macro_f1"],
        "test_loss":            test_eval["loss"],
        "mean_approx_error":    mean_approx_error,
        "mean_energy_retained": mean_energy_retained,
        "mean_param_ratio":     mean_param_ratio,
        **{k: v for k, v in profile.items() if k != "label"},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run SVD low-rank sweep")
    parser.add_argument("--baseline_config", default="configs/baseline.yaml")
    parser.add_argument("--seeds",  nargs="+", type=int,   default=None)
    parser.add_argument("--ratios", nargs="+", type=float, default=None)
    parser.add_argument("--verbose", action="store_true",  default=True)
    args = parser.parse_args()

    with open(args.baseline_config) as f:
        baseline_config = yaml.safe_load(f)

    # Default rank ratios: 25%, 50%, 75%, 100% (reference)
    default_ratios = [0.25, 0.50, 0.75, 1.0]
    seeds  = args.seeds  or baseline_config["seeds"]
    ratios = args.ratios or default_ratios

    results = []
    for seed in seeds:
        for ratio in ratios:
            r = run_low_rank_experiment(
                baseline_config,
                seed=seed,
                rank_ratio=ratio,
                verbose=args.verbose,
            )
            results.append(r)

    # ---- Summary per ratio -------------------------------------------------
    print(f"\n{'='*60}")
    print("  Low-Rank SVD Sweep Summary")
    print(f"  {'Ratio':>8}  {'Acc mean':>10}  {'Acc std':>10}  {'Approx err':>12}")
    for ratio in ratios:
        rows = [r for r in results if r["rank_ratio"] == ratio]
        accs = [r["test_accuracy"]     for r in rows]
        errs = [r["mean_approx_error"] for r in rows]
        print(f"  {ratio*100:>7.0f}%  {np.mean(accs)*100:>9.2f}%  "
              f"{np.std(accs)*100:>9.2f}%  {np.mean(errs):>11.4f}")
    print(f"{'='*60}\n")

    # ---- Save CSV ----------------------------------------------------------
    results_dir = Path(baseline_config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_report_csv(results, str(results_dir / "low_rank_sweep.csv"))


if __name__ == "__main__":
    main()
