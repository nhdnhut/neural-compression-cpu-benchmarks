"""
profiling/reporter.py
=====================
Aggregate all profiling metrics into a single structured dict.

This is the main entry point for profiling a model:
    report = profile_model(model, input_shape=(1,1,28,28))
    # report is a flat dict suitable for writing to CSV
"""

from __future__ import annotations
import os
import json
import csv
from pathlib import Path
import torch
import torch.nn as nn

from profiling.timer  import measure_latency, measure_throughput
from profiling.memory import measure_peak_ram_inference, model_size_mb, model_size_on_disk_mb
from profiling.flops  import count_flops, count_parameters, compute_sparsity


def profile_model(
    model: nn.Module,
    input_shape_single: tuple = (1, 1, 28, 28),
    batch_size_throughput: int = 32,
    latency_warmup: int = 10,
    latency_runs: int = 100,
    saved_path: str | None = None,
    label: str = "model",
) -> dict:
    """
    Run full profiling suite on a model.

    Args:
        model:                  PyTorch model, CPU, eval() mode
        input_shape_single:     shape with batch=1 for latency measurement
        batch_size_throughput:  batch size for throughput measurement
        latency_warmup:         warmup runs for latency timer
        latency_runs:           measurement runs for latency timer
        saved_path:             path to .pth file for on-disk size (optional)
        label:                  name for this configuration

    Returns:
        Flat dict of all metrics, suitable for CSV row.
    """
    model.eval()
    device = "cpu"

    # -- Latency (batch=1) --------------------------------------------------
    dummy_single = torch.zeros(*input_shape_single)
    latency = measure_latency(
        model, dummy_single,
        warmup_runs=latency_warmup,
        measurement_runs=latency_runs,
        device=device,
    )

    # -- Throughput (batch=N) -----------------------------------------------
    input_shape_batch = (batch_size_throughput,) + input_shape_single[1:]
    dummy_batch = torch.zeros(*input_shape_batch)
    throughput = measure_throughput(model, dummy_batch, device=device)

    # -- Peak RAM -----------------------------------------------------------
    mem = measure_peak_ram_inference(model, dummy_single, runs=10)

    # -- FLOPs / MACs -------------------------------------------------------
    flops = count_flops(model, input_shape=input_shape_single)

    # -- Parameters ---------------------------------------------------------
    params = count_parameters(model)

    # -- Sparsity (for pruned models) ---------------------------------------
    sparsity = compute_sparsity(model)

    # -- Model size ---------------------------------------------------------
    size_mb = model_size_mb(model)
    disk_mb = model_size_on_disk_mb(saved_path) if saved_path and os.path.exists(saved_path) else None

    # -- Aggregate ----------------------------------------------------------
    report = {
        "label":                  label,
        # Latency
        "latency_mean_ms":        latency["mean_ms"],
        "latency_std_ms":         latency["std_ms"],
        "latency_p50_ms":         latency["p50_ms"],
        "latency_p95_ms":         latency["p95_ms"],
        # Throughput
        "throughput_samples_sec": throughput["samples_per_sec"],
        # Memory
        "peak_ram_mb":            mem["peak_tracemalloc_mb"],
        "param_bytes_mb":         mem["model_params_mb"],
        "psutil_rss_mb":          mem.get("psutil_rss_mb") if mem.get("psutil_rss_mb") is not None else float("nan"),
        # FLOPs
        "MACs":                   flops["MACs"],
        "FLOPs":                  flops["FLOPs"],
        "MACs_M":                 flops["MACs_M"],
        "FLOPs_M":                flops["FLOPs_M"],
        # Parameters
        "params_trainable":       params["trainable"],
        "params_total":           params["total"],
        # Sparsity
        "global_sparsity":        sparsity["global_sparsity"],
        "total_zeros":            sparsity["total_zeros"],
        # Model size
        "model_size_param_mb":    size_mb,
        "model_size_disk_mb":     disk_mb,
    }
    return report


def save_report_csv(reports: list[dict], filepath: str) -> None:
    """Save a list of profile reports (dicts) to a CSV file."""
    if not reports:
        return
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # Collect union of all keys preserving first-seen order
    fieldnames = []
    for r in reports:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    print(f"[reporter] Saved {len(reports)} rows -> {filepath}")


def print_report(report: dict) -> None:
    """Pretty-print a single report dict."""
    print(f"\n{'-'*55}")
    print(f"  Profile: {report.get('label', 'unknown')}")
    print(f"{'-'*55}")
    print(f"  Latency (batch=1):  {report['latency_mean_ms']:.2f} +/- {report['latency_std_ms']:.2f} ms")
    print(f"  Throughput:         {report['throughput_samples_sec']:.0f} samples/sec")
    print(f"  Peak RAM:           {report['peak_ram_mb']:.3f} MB  (tracemalloc)")
    print(f"  Param size:         {report['model_size_param_mb']:.3f} MB")
    print(f"  Parameters:         {report['params_trainable']:,} trainable")
    print(f"  MACs:               {report['MACs_M']:.2f} M")
    print(f"  Sparsity:           {report['global_sparsity']*100:.1f}%")
    if report.get("model_size_disk_mb"):
        print(f"  On-disk size:       {report['model_size_disk_mb']:.3f} MB")
    print(f"{'-'*55}\n")
