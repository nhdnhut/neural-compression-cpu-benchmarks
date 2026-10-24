"""
report/generate_tables.py
==========================
Read benchmark_all.csv and generate LaTeX table fragments with real numbers.

Usage:
    python report/generate_tables.py
    python report/generate_tables.py --csv benchmarks/results/benchmark_all.csv

Outputs:
    report/sections/full_results_table.tex  -- Appendix full table
    report/sections/results_macros.tex      -- \\RESULTFILE{} definitions

After running, rebuild the paper:
    cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_acc(mean: float, std: float) -> str:
    r"""Format accuracy as 'XX.XX ± Y.YY\%'"""
    return f"{mean*100:.2f} $\\pm$ {std*100:.2f}\\%"


def fmt_val(mean: float, std: float, decimals: int = 2) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def fmt_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.2f}\\%"


def agg(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Return (mean, std) of column, ignoring NaN."""
    vals = df[col].dropna().astype(float)
    return float(vals.mean()), float(vals.std(ddof=0))


# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = df.columns.str.strip()
    return df


def make_macros(df: pd.DataFrame, baseline_acc: float) -> dict[str, str]:
    """Return a dict: macro_name -> LaTeX value string."""
    macros = {}

    # Baseline
    base = df[df["method"] == "baseline"]
    m, s = agg(base, "test_accuracy")
    macros["baseline_acc"]  = fmt_acc(m, s)
    macros["baseline_f1"]   = fmt_acc(*agg(base, "test_macro_f1"))
    macros["baseline_lat"]  = fmt_val(*agg(base, "latency_mean_ms"))
    macros["baseline_size"] = fmt_val(*agg(base, "model_size_param_mb"), decimals=3)
    macros["baseline_macs"] = f"{agg(base, 'MACs_M')[0]:.3f}M"
    base_acc = agg(base, "test_accuracy")[0]

    # Pruning
    for ratio, tag in [(0.0, "p0"), (0.25, "p25"), (0.50, "p50"),
                       (0.75, "p75"), (0.90, "p90")]:
        rows = df[(df["method"] == "pruning") & (df["pruning_ratio"] == ratio)]
        if ratio == 0.0:
            rows = df[df["method"] == "baseline"]
        if rows.empty:
            continue
        m_a, s_a = agg(rows, "test_accuracy")
        m_f, s_f = agg(rows, "test_macro_f1")
        m_l, s_l = agg(rows, "latency_mean_ms")
        delta_acc = m_a - base_acc
        macros[f"{tag}_acc"]  = fmt_acc(m_a, s_a)
        macros[f"{tag}_dacc"] = fmt_delta(delta_acc)
        macros[f"{tag}_f1"]   = fmt_acc(m_f, s_f)
        macros[f"{tag}_lat"]  = fmt_val(m_l, s_l)
        if tag != "p0":
            base_lat = agg(df[df["method"] == "baseline"], "latency_mean_ms")[0]
            macros[f"{tag}_dlat"] = fmt_delta((m_l - base_lat) / base_lat)

    # Quantization
    for method, tag in [("fp32", "q_fp32"), ("dynamic_int8", "q_int8")]:
        rows = df[(df["method"] == "quantization") & (df["quantization"] == method)]
        if rows.empty:
            # fallback: fp32 quant reference = baseline
            if method == "fp32":
                rows = df[df["method"] == "baseline"]
        if rows.empty:
            continue
        m_a, s_a = agg(rows, "test_accuracy")
        m_s, s_s = agg(rows, "model_size_param_mb")
        m_l, s_l = agg(rows, "latency_mean_ms")
        macros[f"{tag}_acc"]  = fmt_acc(m_a, s_a)
        macros[f"{tag}_dacc"] = fmt_delta(m_a - base_acc)
        macros[f"{tag}_size"] = fmt_val(m_s, s_s, decimals=3)
        macros[f"{tag}_lat"]  = fmt_val(m_l, s_l)

    # Low-rank
    for ratio, tag in [(1.0, "lr100"), (0.75, "lr75"), (0.50, "lr50"), (0.25, "lr25")]:
        rows = df[(df["method"] == "low_rank") & (df["rank_ratio"] == ratio)]
        if ratio == 1.0:
            rows = df[df["method"] == "baseline"]
        if rows.empty:
            continue
        m_a, s_a = agg(rows, "test_accuracy")
        macros[f"{tag}_acc"]  = fmt_acc(m_a, s_a)
        macros[f"{tag}_dacc"] = fmt_delta(m_a - base_acc)
        if "mean_energy_retained" in rows.columns and ratio < 1.0:
            m_e, _ = agg(rows, "mean_energy_retained")
            macros[f"{tag}_eta"]  = f"{m_e*100:.1f}\\%"
            m_fe, _ = agg(rows, "mean_approx_error")
            macros[f"{tag}_ferr"] = f"{m_fe:.4f}"

    return macros


# ---------------------------------------------------------------------------
# Output: results_macros.tex (\\newcommand{\\RESULTFILE{name}}{value})
# ---------------------------------------------------------------------------

def write_macros_tex(macros: dict[str, str], out_path: Path) -> None:
    lines = [
        "% Auto-generated by report/generate_tables.py — DO NOT EDIT",
        "% Include in main.tex preamble: \\input{sections/results_macros}",
        "% Then use \\RESULTFILE{key} in sections.",
        "",
        "% Redefine \\RESULTFILE to expand the macro by name:",
        "\\newcommand{\\RESULTFILE}[1]{\\csname RESULT@#1\\endcsname}",
        "",
    ]
    for k, v in macros.items():
        # Store as \RESULT@key
        lines.append(f"\\expandafter\\newcommand\\csname RESULT@{k}\\endcsname{{{v}}}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[generate_tables] Written {out_path}")


# ---------------------------------------------------------------------------
# Output: full_results_table.tex
# ---------------------------------------------------------------------------

def write_full_table(df: pd.DataFrame, out_path: Path) -> None:
    """Generate a full LaTeX longtable of all results."""
    cols_display = [
        ("method", "Method"),
        ("pruning_ratio", "Sparsity"),
        ("quantization", "Quant."),
        ("rank_ratio", "Rank"),
        ("seed", "Seed"),
        ("test_accuracy", "Acc."),
        ("test_macro_f1", "F1"),
        ("latency_mean_ms", "Lat (ms)"),
        ("model_size_param_mb", "Size (MB)"),
        ("MACs_M", "MACs (M)"),
        ("global_sparsity", "Sparsity"),
    ]
    available = [(c, h) for c, h in cols_display if c in df.columns]
    col_keys   = [c for c, _ in available]
    col_headers = [h for _, h in available]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Complete benchmark results across all methods and seeds.}",
        "\\label{tab:full_results}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{" + "l" * len(available) + "}",
        "\\toprule",
        " & ".join(col_headers) + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        cells = []
        for col in col_keys:
            v = row.get(col, "")
            if pd.isna(v):
                cells.append("--")
            elif isinstance(v, float):
                if col in ("test_accuracy", "test_macro_f1", "global_sparsity"):
                    cells.append(f"{v*100:.2f}\\%")
                elif col in ("latency_mean_ms", "model_size_param_mb", "MACs_M"):
                    cells.append(f"{v:.3f}")
                elif col in ("pruning_ratio", "rank_ratio"):
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:.4f}")
            else:
                cells.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[generate_tables] Written {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(ROOT / "benchmarks" / "results" / "benchmark_all.csv"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "report" / "sections"),
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[generate_tables] ERROR: {csv_path} not found.")
        print("  Run: python benchmarks/run_all.py first.")
        sys.exit(1)

    df = load_results(str(csv_path))
    print(f"[generate_tables] Loaded {len(df)} rows from {csv_path}")
    print(f"  Methods: {sorted(df['method'].unique())}")

    base_acc = df[df["method"] == "baseline"]["test_accuracy"].mean()
    macros = make_macros(df, base_acc)
    print(f"[generate_tables] Generated {len(macros)} macros")

    write_macros_tex(macros, out_dir / "results_macros.tex")
    write_full_table(df, out_dir / "full_results_table.tex")

    print("[generate_tables] Done. Rebuild PDF:")
    print("  cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex")


if __name__ == "__main__":
    main()
