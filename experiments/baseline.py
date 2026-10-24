"""
experiments/baseline.py
=======================
Run the baseline experiment (E00):
  Train SmallCNN on Fashion-MNIST with 3 seeds.
  Profile the trained model.
  Save results to benchmarks/results/baseline.csv

Usage:
    python experiments/baseline.py
    python experiments/baseline.py --config configs/baseline.yaml --seeds 42 123 7
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Make sure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nn.architectures import SmallCNN
from data.loaders import load_fashion_mnist
from optim.trainer import train, evaluate, set_seed
from profiling.reporter import profile_model, save_report_csv, print_report


def run_baseline(config: dict, seed: int, verbose: bool = True) -> dict:
    """
    Train baseline SmallCNN on Fashion-MNIST for one seed.

    Returns:
        dict with training history, evaluation results, and profiling report.
    """
    print(f"\n{'='*60}")
    print(f"  Baseline experiment | seed={seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # ---- Data ---------------------------------------------------------------
    ds_cfg = config["dataset"]
    data   = load_fashion_mnist(
        data_dir     = ds_cfg["data_dir"],
        batch_size   = ds_cfg["batch_size"],
        val_fraction = ds_cfg["val_fraction"],
        seed         = seed,
    )

    # ---- Model --------------------------------------------------------------
    m_cfg = config["model"]
    model = SmallCNN(
        num_classes = m_cfg["num_classes"],
        dropout_p   = m_cfg["dropout_p"],
    )
    print(f"  Parameters: {model.param_count():,}")

    # ---- Training -----------------------------------------------------------
    t_cfg = config["training"]
    t0    = time.perf_counter()
    history = train(
        model        = model,
        train_loader = data["train_loader"],
        val_loader   = data["val_loader"],
        config       = t_cfg,
        verbose      = verbose,
    )
    training_time_sec = time.perf_counter() - t0
    print(f"  Training time: {training_time_sec:.1f}s")

    # ---- Evaluation on test set (done ONCE, after training) ----------------
    criterion = torch.nn.CrossEntropyLoss()
    test_eval = evaluate(
        model       = model,
        loader      = data["test_loader"],
        criterion   = criterion,
        class_names = data["class_names"],
    )
    print(f"  Test accuracy: {test_eval['accuracy']*100:.2f}%  |  Macro F1: {test_eval['macro_f1']*100:.2f}%")

    # ---- Save model ---------------------------------------------------------
    models_dir = Path(config["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(models_dir / f"baseline_seed{seed}.pth")
    torch.save(model.state_dict(), model_path)

    # ---- Profile ------------------------------------------------------------
    p_cfg = config["profiling"]
    profile = profile_model(
        model                  = model.eval(),
        input_shape_single     = tuple(p_cfg["input_shape"]),
        batch_size_throughput  = p_cfg["batch_size_throughput"],
        latency_warmup         = p_cfg["latency_warmup"],
        latency_runs           = p_cfg["latency_runs"],
        saved_path             = model_path,
        label                  = f"baseline_seed{seed}",
    )
    print_report(profile)

    # ---- Combine results ----------------------------------------------------
    result = {
        # Identifiers
        "exp_id":            "E00",
        "method":            "baseline",
        "pruning_ratio":     0.0,
        "quantization":      "fp32",
        "rank_ratio":        1.0,
        "seed":              seed,
        "dataset":           "FashionMNIST",
        # Training
        "epochs":            t_cfg["epochs"],
        "training_time_sec": training_time_sec,
        "best_val_acc":      history["best_val_acc"],
        # Evaluation
        "test_accuracy":     test_eval["accuracy"],
        "test_macro_f1":     test_eval["macro_f1"],
        "test_loss":         test_eval["loss"],
        # Profile (flattened)
        **{k: v for k, v in profile.items() if k != "label"},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument("--config",  default="configs/baseline.yaml")
    parser.add_argument("--seeds",   nargs="+", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seeds   = args.seeds or config["seeds"]
    results = []

    for seed in seeds:
        result = run_baseline(config, seed=seed, verbose=args.verbose)
        results.append(result)

    # ---- Aggregate and report ----------------------------------------------
    accs = [r["test_accuracy"] for r in results]
    f1s  = [r["test_macro_f1"]  for r in results]
    print(f"\n{'='*60}")
    print(f"  Baseline Results (n={len(seeds)} seeds)")
    print(f"  Accuracy:  {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%")
    print(f"  Macro F1:  {np.mean(f1s)*100:.2f} +/- {np.std(f1s)*100:.2f}%")
    print(f"{'='*60}\n")

    # ---- Save CSV ----------------------------------------------------------
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(results_dir / "baseline.csv")
    save_report_csv(results, csv_path)

    # ---- Save mean +/- std summary ------------------------------------------
    summary = {
        "exp_id":           "E00",
        "method":           "baseline",
        "n_seeds":          len(seeds),
        "test_accuracy_mean": float(np.mean(accs)),
        "test_accuracy_std":  float(np.std(accs)),
        "test_macro_f1_mean": float(np.mean(f1s)),
        "test_macro_f1_std":  float(np.std(f1s)),
        "latency_mean_ms_mean": float(np.mean([r["latency_mean_ms"] for r in results])),
        "params_trainable": results[0]["params_trainable"],
        "MACs_M": results[0]["MACs_M"],
        "model_size_param_mb": results[0]["model_size_param_mb"],
    }
    summary_path = str(results_dir / "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
