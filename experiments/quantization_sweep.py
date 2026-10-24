"""
experiments/quantization_sweep.py
==================================
Run quantization experiments (E05):
  Load baseline model, apply dynamic INT8 PTQ, evaluate, profile.

Usage:
    python experiments/quantization_sweep.py
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
from compress.quantization import apply_dynamic_ptq
from profiling.reporter import profile_model, save_report_csv, print_report


def run_quantization_experiment(
    baseline_config: dict,
    seed: int,
    method: str = "dynamic_int8",
) -> dict:
    """
    Quantize the baseline model and evaluate.
    method: 'fp32' (no quantization, for reference) or 'dynamic_int8'
    """
    exp_label = f"quant_{method}_seed{seed}"
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

    # ---- Apply quantization ------------------------------------------------
    if method == "dynamic_int8":
        model = apply_dynamic_ptq(model, dtype=torch.qint8, copy_model=True)
        print("  Applied Dynamic PTQ INT8 (Linear layers quantized)")
    elif method == "fp32":
        print("  FP32 reference (no quantization)")
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # ---- Evaluate ----------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    # NOTE: quantized models may not support CrossEntropyLoss directly
    # We evaluate the quantized model's outputs using float criterion
    try:
        test_eval = evaluate(model, data["test_loader"], criterion)
    except Exception as e:
        print(f"  [Warning] Evaluation error: {e}")
        print("  Trying with dequantized outputs...")
        # Fallback: wrap model outputs
        test_eval = _evaluate_quantized(model, data["test_loader"])

    print(f"  Test accuracy: {test_eval['accuracy']*100:.2f}%  |  F1: {test_eval['macro_f1']*100:.2f}%")

    # ---- Save quantized model ----------------------------------------------
    quant_path = str(models_dir / f"quantized_{method}_seed{seed}.pth")
    torch.save(model.state_dict(), quant_path)

    # ---- Profile -----------------------------------------------------------
    p_cfg = baseline_config["profiling"]
    profile = profile_model(
        model              = model.eval(),
        input_shape_single = tuple(p_cfg["input_shape"]),
        batch_size_throughput  = p_cfg["batch_size_throughput"],
        latency_warmup     = p_cfg["latency_warmup"],
        latency_runs       = p_cfg["latency_runs"],
        saved_path         = quant_path,
        label              = exp_label,
    )
    print_report(profile)

    result = {
        "exp_id":         "E05",
        "method":         "quantization",
        "pruning_ratio":  0.0,
        "quantization":   method,
        "rank_ratio":     1.0,
        "seed":           seed,
        "dataset":        "FashionMNIST",
        "test_accuracy":  test_eval["accuracy"],
        "test_macro_f1":  test_eval["macro_f1"],
        "test_loss":      test_eval.get("loss", float("nan")),
        **{k: v for k, v in profile.items() if k != "label"},
    }
    return result


def _evaluate_quantized(model, loader):
    """Fallback evaluation for quantized models."""
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "accuracy":   (all_preds == all_labels).mean(),
        "macro_f1":   f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "loss":       float("nan"),   # not computed in this fallback
    }


def main():
    parser = argparse.ArgumentParser(description="Run quantization sweep")
    parser.add_argument("--baseline_config", default="configs/baseline.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    with open(args.baseline_config) as f:
        baseline_config = yaml.safe_load(f)

    seeds   = args.seeds or baseline_config["seeds"]
    methods = ["fp32", "dynamic_int8"]
    results = []

    for seed in seeds:
        for method in methods:
            r = run_quantization_experiment(baseline_config, seed=seed, method=method)
            results.append(r)

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Quantization Summary")
    for method in methods:
        rows = [r for r in results if r["quantization"] == method]
        accs = [r["test_accuracy"] for r in rows]
        lats = [r["latency_mean_ms"] for r in rows]
        sizes = [r["model_size_param_mb"] for r in rows]
        print(f"  {method:>15}  acc={np.mean(accs)*100:.2f}+/-{np.std(accs)*100:.2f}%  "
              f"lat={np.mean(lats):.2f}ms  size={np.mean(sizes):.3f}MB")
    print(f"{'='*60}\n")

    results_dir = Path(baseline_config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_report_csv(results, str(results_dir / "quantization_sweep.csv"))


if __name__ == "__main__":
    main()

