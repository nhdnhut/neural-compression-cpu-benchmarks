"""
experiments/pruning_sweep.py
============================
Run pruning experiments (E01–E04):
  Load baseline model, apply magnitude pruning at multiple ratios,
  fine-tune, evaluate, profile, save results.

Usage:
    python experiments/pruning_sweep.py
    python experiments/pruning_sweep.py --seeds 42 --ratios 0.0 0.5 0.9
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn.architectures import SmallCNN
from data.loaders import load_fashion_mnist
from optim.trainer import train, evaluate, set_seed
from compress.pruning import apply_magnitude_pruning, make_pruning_permanent, _compute_sparsity_stats
from profiling.reporter import profile_model, save_report_csv, print_report


def run_pruning_experiment(
    config: dict,
    baseline_config: dict,
    seed: int,
    pruning_ratio: float,
    verbose: bool = True,
) -> dict:
    """
    Load baseline model, prune, fine-tune, evaluate, profile.
    Returns a results dict (one row of the benchmark table).
    """
    exp_label = f"pruning_{int(pruning_ratio*100)}pct_seed{seed}"
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
    models_dir = Path(baseline_config["output"]["models_dir"])
    baseline_path = str(models_dir / f"baseline_seed{seed}.pth")

    model = SmallCNN(
        num_classes = baseline_config["model"]["num_classes"],
        dropout_p   = baseline_config["model"]["dropout_p"],
    )
    state = torch.load(baseline_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded baseline from {baseline_path}")

    # ---- Apply pruning ------------------------------------------------------
    sparsity_stats = apply_magnitude_pruning(
        model         = model,
        pruning_ratio = pruning_ratio,
        make_permanent= False if pruning_ratio > 0 else True,
    )
    print(f"  Target pruning ratio: {pruning_ratio*100:.1f}%")

    # ---- Fine-tune ----------------------------------------------------------
    p_cfg = config["pruning"]
    ft_epochs = p_cfg["finetune_epochs"] if pruning_ratio > 0 else 0

    if ft_epochs > 0:
        ft_config = {**baseline_config["training"]}
        ft_config["epochs"] = ft_epochs
        ft_config["lr"] = baseline_config["training"]["lr"] * p_cfg["finetune_lr_factor"]
        ft_config["scheduler"] = None   # no scheduler for short finetune

        print(f"  Fine-tuning for {ft_epochs} epochs (lr={ft_config['lr']:.1e})...")
        history = train(model, data["train_loader"], data["val_loader"],
                        ft_config, verbose=verbose)
    else:
        history = {"best_val_acc": 0.0}

    # Make permanent AFTER fine-tuning so weights stay zeroed
    if pruning_ratio > 0:
        make_pruning_permanent(model)

    sparsity_stats = _compute_sparsity_stats(model)
    actual_sparsity = sparsity_stats["global_sparsity"]
    print(f"  Final verified global sparsity: {actual_sparsity*100:.1f}%")

    # ---- Evaluate -----------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    test_eval = evaluate(model, data["test_loader"], criterion)
    print(f"  Test accuracy: {test_eval['accuracy']*100:.2f}%  |  F1: {test_eval['macro_f1']*100:.2f}%")

    # ---- Save pruned model --------------------------------------------------
    pruned_path = str(models_dir / f"pruned_{int(pruning_ratio*100)}pct_seed{seed}.pth")
    torch.save(model.state_dict(), pruned_path)

    # ---- Profile ------------------------------------------------------------
    p_profile_cfg = baseline_config["profiling"]
    profile = profile_model(
        model              = model.eval(),
        input_shape_single = tuple(p_profile_cfg["input_shape"]),
        batch_size_throughput  = p_profile_cfg["batch_size_throughput"],
        latency_warmup     = p_profile_cfg["latency_warmup"],
        latency_runs       = p_profile_cfg["latency_runs"],
        saved_path         = pruned_path,
        label              = exp_label,
    )
    print_report(profile)

    result = {
        "exp_id":              f"E_prune{int(pruning_ratio * 100):03d}",
        "method":              "pruning",
        "pruning_ratio":       pruning_ratio,
        "actual_sparsity":     actual_sparsity,
        "quantization":        "fp32",
        "rank_ratio":          1.0,
        "seed":                seed,
        "dataset":             "FashionMNIST",
        "finetune_epochs":     ft_epochs,
        "test_accuracy":       test_eval["accuracy"],
        "test_macro_f1":       test_eval["macro_f1"],
        "test_loss":           test_eval["loss"],
        **{k: v for k, v in profile.items() if k != "label"},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run pruning sweep")
    parser.add_argument("--config",          default="configs/pruning.yaml")
    parser.add_argument("--baseline_config", default="configs/baseline.yaml")
    parser.add_argument("--seeds",    nargs="+", type=int, default=None)
    parser.add_argument("--ratios",   nargs="+", type=float, default=None)
    parser.add_argument("--finetune_epochs", type=int, default=None)
    parser.add_argument("--verbose",  action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.baseline_config) as f:
        baseline_config = yaml.safe_load(f)

    if args.finetune_epochs is not None:
        config["pruning"]["finetune_epochs"] = args.finetune_epochs

    seeds   = args.seeds  or config["seeds"]
    ratios  = args.ratios or config["pruning"]["ratios"]
    results = []

    for seed in seeds:
        for ratio in ratios:
            r = run_pruning_experiment(
                config, baseline_config, seed=seed,
                pruning_ratio=ratio, verbose=args.verbose,
            )
            results.append(r)

    # ---- Summary per ratio -------------------------------------------------
    print(f"\n{'='*60}")
    print("  Pruning Sweep Summary")
    print(f"  {'Ratio':>8}  {'Acc mean':>10}  {'Acc std':>10}  {'Sparsity':>10}")
    for ratio in ratios:
        rows = [r for r in results if r["pruning_ratio"] == ratio]
        accs = [r["test_accuracy"] for r in rows]
        sp   = [r["actual_sparsity"] for r in rows]
        print(f"  {ratio*100:>7.0f}%  {np.mean(accs)*100:>9.2f}%  "
              f"{np.std(accs)*100:>9.2f}%  {np.mean(sp)*100:>9.1f}%")
    print(f"{'='*60}\n")

    # ---- Save CSV ----------------------------------------------------------
    results_dir = Path(baseline_config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_report_csv(results, str(results_dir / "pruning_sweep.csv"))


if __name__ == "__main__":
    main()

