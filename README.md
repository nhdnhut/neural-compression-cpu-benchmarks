# An Empirical Study of Accuracyâ€“Resource Trade-offs in Neural Network Compression under CPU-Only Constraints

> A rigorous empirical benchmark of compression methods on small neural networks, measured on CPU-only hardware.
> PyTorch | Pruning | INT8 Dynamic Quantization | SVD Decomposition | CPU Benchmark Harness

[![Python](https://img.shields.io/badge/Python-3.11+-brightgreen?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-32%20Passed-brightgreen?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

---

## Research Questions

| RQ | Question |
|---|---|
| **RQ1** | How does model quality relate to parameter count? |
| **RQ2** | Does FLOPs predict actual CPU inference latency? |
| **RQ3** | What is the accuracyâ€“resource Pareto trade-off across compression strategies? |
| **RQ4** | At what pruning ratio does accuracy degradation become non-linear? |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests (verify backprop is correct)
pytest tests/ -v

# 3. Run full benchmark (downloads Fashion-MNIST automatically)
python benchmarks/run_all.py

# 4. Or run individual phases:
python experiments/baseline.py
python experiments/pruning_sweep.py
python experiments/quantization_sweep.py
```

---

## Repository Structure

```
efficient-nn-benchmark/
â”‚
â”œâ”€â”€ core/               NumPy mini-framework (educational)
â”‚   â”œâ”€â”€ layers.py       Linear, ReLU, Sigmoid, Tanh, Dropout, Flatten
â”‚   â”œâ”€â”€ losses.py       CrossEntropyLoss, MSELoss (numerically stable)
â”‚   â”œâ”€â”€ optimizers.py   SGD (with momentum), Adam (bias-corrected)
â”‚   â”œâ”€â”€ mlp.py          Sequential MLP container
â”‚   â””â”€â”€ trainer.py      Mini-batch training loop
â”‚
â”œâ”€â”€ nn/                 PyTorch models
â”‚   â””â”€â”€ architectures.py  SmallCNN (~207K params), SmallMLP (tabular)
â”‚
â”œâ”€â”€ optim/
â”‚   â””â”€â”€ trainer.py      PyTorch training engine with F1 evaluation
â”‚
â”œâ”€â”€ data/
â”‚   â””â”€â”€ loaders.py      Fashion-MNIST + UCI Dry Bean loaders
â”‚
â”œâ”€â”€ compression/
â”‚   â”œâ”€â”€ pruning.py      Magnitude unstructured pruning (Han et al. 2015)
â”‚   â”œâ”€â”€ quantization.py Dynamic PTQ INT8 (torch.quantization)
â”‚   â””â”€â”€ low_rank.py     SVD low-rank approximation (Denton et al. 2014)
â”‚
â”œâ”€â”€ profiling/
â”‚   â”œâ”€â”€ timer.py        Wall-clock latency (warmup + percentiles)
â”‚   â”œâ”€â”€ memory.py       Peak RAM (tracemalloc + psutil)
â”‚   â”œâ”€â”€ flops.py        FLOPs/MACs (thop) + sparsity analysis
â”‚   â””â”€â”€ reporter.py     Aggregate all metrics â†’ flat CSV row
â”‚
â”œâ”€â”€ experiments/
â”‚   â”œâ”€â”€ baseline.py        E00: SmallCNN baseline Ã— 3 seeds
â”‚   â”œâ”€â”€ pruning_sweep.py   E01â€“E04: pruning at 25/50/75/90%
â”‚   â””â”€â”€ quantization_sweep.py  E05: FP32 vs Dynamic INT8
â”‚
â”œâ”€â”€ benchmarks/
â”‚   â”œâ”€â”€ run_all.py         Master script: runs all experiments
â”‚   â””â”€â”€ results/           Raw CSV outputs (committed)
â”‚
â”œâ”€â”€ visualization/
â”‚   â””â”€â”€ pareto.py          Pareto frontier + dashboard plots
â”‚
â”œâ”€â”€ configs/
â”‚   â”œâ”€â”€ baseline.yaml
â”‚   â”œâ”€â”€ pruning.yaml
â”‚   â””â”€â”€ quantization.yaml
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ test_layers.py     Numerical gradient checks (must pass!)
â”‚   â”œâ”€â”€ test_losses.py     Loss function correctness
â”‚   â””â”€â”€ test_optimizers.py Optimizer convergence tests
â”‚
â”œâ”€â”€ notebooks/             Analysis notebooks (Jupyter)
â””â”€â”€ report/                Written report + figures
```

---

## Experimental Design

| Exp | Method | Pruning | Precision | Seeds |
|---|---|---|---|---|
| E00 | Baseline | 0% | FP32 | 3 |
| E01 | Pruning | 25% | FP32 | 3 |
| E02 | Pruning | 50% | FP32 | 3 |
| E03 | Pruning | 75% | FP32 | 3 |
| E04 | Pruning | 90% | FP32 | 3 |
| E05 | Quantization | 0% | INT8 | 3 |

Total: **18 runs** for MVRP. Results reported as **mean Â± std**.

---

## Metrics

| Category | Metrics |
|---|---|
| Model quality | Test accuracy, Macro F1, Per-class F1 |
| Theoretical cost | Parameters, FLOPs (M), MACs (M) |
| Actual hardware cost | Latency ms (batch=1), Throughput (batch=32), Peak RAM (MB) |
| Size | Parameter tensor MB, On-disk .pth MB |
| Compression | Sparsity %, Compression ratio |

---

## Design Principles

1. **One variable at a time**: each experiment changes one compression parameter only.
2. **Test set held out**: hyperparameters tuned on val set, test set evaluated once at the end.
3. **3 seeds minimum**: all results reported as mean Â± std.
4. **Reproducible configs**: all hyperparameters in YAML files, committed to repo.
5. **Theoretical vs actual**: FLOPs and latency reported separately â€” both matter.

---

## Honest Contribution Statement

This project is a **reproducible empirical benchmark**, not a novel compression algorithm.

**Certain contributions:**
- Systematic benchmark of pruning + quantization on CPU, small-model regime
- Open-source framework with NumPy mini-engine + compression utilities
- FLOPs-vs-latency comparison on CPU (documents the mismatch)

**Not claimed:**
- Novel compression methods
- State-of-the-art results
- Generalisation beyond Fashion-MNIST / SmallCNN

---

## Literature

| Paper | Year | Relevance |
|---|---|---|
| Han et al., NIPS | 2015 | Magnitude pruning (reproduced here) |
| Jacob et al., CVPR | 2018 | INT8 quantization foundations |
| Frankle & Carlin, ICLR | 2019 | Lottery ticket hypothesis (context) |
| Menghani, ACM CSUR | 2023 | Survey: efficient DL (read first) |

---

## Definition of Done

- [x] `pytest tests/ -v` passes (32/32 tests, gradient checks < 1e-5)
- [x] Baseline trained and profiled on Fashion-MNIST (92.37% test accuracy)
- [x] Pruning sweep complete across 5 ratios (0%, 25%, 50%, 75%, 90%) with verified sparsity retention
- [x] Quantization sweep complete (FP32 vs Dynamic INT8 PTQ)
- [x] All metrics profiled on CPU: Accuracy, F1, Latency (mean/p50/p95), RAM, FLOPs, Size
- [x] Publication Pareto plots generated in `benchmarks/figures/`:
  - `pareto_accuracy_vs_size.png`
  - `pareto_accuracy_vs_latency.png`
  - `pareto_accuracy_vs_ram.png`
  - `pareto_dashboard.png` (6-panel trade-off dashboard)
  - `accuracy_vs_compression_ratio.png`
- [ ] Multi-seed expansion (seeds 123, 7) for full confidence intervals
- [ ] Written technical report

---

## Benchmark Results (Fashion-MNIST / SmallCNN on CPU)

| Experiment | Method | Pruning | Format | Sparsity | Test Acc | Macro F1 | Latency (ms) | Peak RAM (MB) | Disk Size (MB) |
|---|---|---|---|---|---|---|---|---|---|
| **E00** | Baseline | 0% | FP32 | 0.0% | **92.37%** | 92.35% | 0.56 | 0.001 | 0.793 |
| **E02** | Pruning | 25% | FP32 | 25.0% | **92.14%** | 92.13% | 0.64 | 0.001 | 0.793 |
| **E03** | Pruning | 50% | FP32 | 50.0% | **92.09%** | 92.08% | 0.80 | 0.001 | 0.793 |
| **E04** | Pruning | 75% | FP32 | 75.0% | **92.28%** | 92.27% | 0.69 | 0.001 | 0.793 |
| **E05** | Pruning | 90% | FP32 | 90.0% | **91.61%** | 91.60% | 0.34 | 0.001 | 0.793 |
| **E05** | Quantization | 0% | Dynamic INT8 | 0.0% | **92.36%** | 92.33% | 0.69 | 0.002 | **0.217** (3.65x) |



---

## Author and License

- Author: Nhut Nguyen (@nhdnhut)
- License: MIT License (see LICENSE)

